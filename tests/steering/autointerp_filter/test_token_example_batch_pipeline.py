from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.eval.autointerp_filter.token_examples as token_examples_module
from steering.eval.autointerp_filter.batch_validation import (
    BatchCalibrationContract,
    BatchToleranceProfile,
    NumericTolerance,
)
from steering.eval.autointerp_filter.batching import (
    MaterializationPhase,
    PreparedBatch,
    PreparedDoc,
    TargetBlock,
    target_plan_hash,
)
from steering.eval.autointerp_filter.cache import TokenExampleCache
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    SourceIndexConfig,
    TokenExamplesConfig,
    validate_config,
)
from steering.eval.autointerp_filter.contracts import runner_manifest_token_examples
from steering.eval.autointerp_filter.document_blacklist import resolve_document_blacklist
from steering.eval.autointerp_filter.intruder import IntruderDetectionJudge
from steering.eval.autointerp_filter.runner import AutoInterpFilterRunner
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    TokenExampleSampler,
    _BlockBuild,
    _DocActivations,
)
from steering.eval.autointerp_filter.types import (
    AutoInterpBatchValidationError,
    AutoInterpConfigError,
    AutoInterpDataError,
    IntruderTrial,
    IntruderTrialResult,
    ScoringTarget,
    SourceDocRef,
    TargetKind,
    TokenExampleProvider,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import PositiveInt


def _target(feature_id: int) -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=f"feature:{feature_id}",
        display_name=f"feature_{feature_id}",
    )


def _doc_ref(feature_id: int, index: int, *, positive: bool) -> SourceDocRef:
    return SourceDocRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=feature_id * 1000 + (index if positive else 500 + index),
        segment_token_start=0,
        segment_token_end=4,
    )


def _window(target: ScoringTarget, *, marker: int, active: bool) -> TokenWindowRef:
    values = (0.0, float(marker + 1), 0.0, 0.0) if active else (0.0,) * 4
    return TokenWindowRef(
        shard_stem="fixture",
        shard_index=0,
        doc_id=marker,
        source_text_hash=f"hash-{marker}",
        token_start=0,
        token_end=4,
        token_ids=(marker * 10, marker * 10 + 1, marker * 10 + 2, marker * 10 + 3),
        token_strings=("a", "b", "c", "d"),
        target_feature_ids=target.feature_ids,
        activations=values,
        max_activation=max(values),
        quantile=None,
        active=active,
    )


def _sentinel_examples(target: ScoringTarget, marker: int) -> TokenExampleSet:
    return TokenExampleSet(
        positives=(_window(target, marker=marker, active=True),),
        non_activating=(_window(target, marker=marker + 100, active=False),),
    )


def _config(
    tmp_path: Path,
    *,
    batch_size: int = 2,
) -> AutoInterpFilterConfig:
    output = tmp_path / f"out-b{batch_size}"
    return AutoInterpFilterConfig(
        feature_dir="root/s4/l1/a_res/p_max_pool/f16k/m_gemma-3-4b-pt/base/context_independent/no_bos",
        output_dir=str(output),
        n_sae_features=PositiveInt.parse(32),
        token_example_cache_path=str(output / "token_examples.sqlite"),
        judge_cache_path=str(output / "judge.sqlite"),
        target_scores_path=str(output / "scores.jsonl"),
        ranked_yaml_path=str(output / "ranked.yaml"),
        accepted_features_path=str(output / "accepted.txt"),
        manual_feature_filter_path=str(output / "manual.yaml"),
        manual_feature_filter_meta_path=str(output / "manual.json"),
        report_output_path=str(output / "report.json"),
        trial_audit_path=str(output / "audit.jsonl"),
        trial_audit_yaml_path=str(output / "audit.yaml"),
        runner_manifest_path=str(output / "manifest.json"),
        lockfile_path=str(output / ".lockfile"),
        token_examples=TokenExamplesConfig(
            device="cuda",
            model="gemma-3-4b-pt",
            activation_backend="native",
            model_dtype="bfloat16",
            example_ctx_len=PositiveInt.parse(4),
            max_positive_source_docs_per_target=PositiveInt.parse(4),
            max_intruder_source_docs=PositiveInt.parse(16),
            max_windows_per_source_doc=PositiveInt.parse(2),
            max_positive_windows_per_target=PositiveInt.parse(3),
            max_non_activating_windows_per_target=PositiveInt.parse(3),
            materialization_batch_size=PositiveInt.parse(batch_size),
        ),
        source_index=SourceIndexConfig(
            path=str(output / "source.sqlite"),
            manifest_path=str(output / "source.manifest.json"),
        ),
    )


def _profile(batch_size: int) -> BatchToleranceProfile:
    return BatchToleranceProfile(
        name=f"test-batch-{batch_size}",
        batch_size=batch_size,
        selected_sae=NumericTolerance(max_abs=0.0, p99_abs=0.0, mean_abs=0.0),
        raw_residual=NumericTolerance(max_abs=0.0, p99_abs=0.0, mean_abs=0.0),
        calibration_contract=BatchCalibrationContract(
            model_name="gemma-3-4b-pt",
            model_dtype="bfloat16",
            activation_type="residual",
            layer=1,
            segment_size=4,
            sae_release="test-release",
            sae_id="test-sae",
            hook_name="blocks.1.hook_resid_post",
        ),
        max_support_mismatches=0,
    )


def _harness_provider(
    tmp_path: Path,
    targets: list[ScoringTarget],
    *,
    batch_size: int = 2,
    prefetch: bool = True,
    positive_docs: int = 2,
    intruder_docs: int = 3,
) -> tuple[CachedTokenExampleProvider, list[int], list[str]]:
    provider = object.__new__(CachedTokenExampleProvider)
    provider._cfg = _config(tmp_path, batch_size=batch_size)
    provider._feature_dir = Path(provider._cfg.feature_dir)
    provider._n_sae_features = PositiveInt.parse(32)
    provider._targets = list(targets)
    provider._prepared = True
    provider._target_plan_hash = target_plan_hash(targets)
    provider._feature_fingerprint = "feature-fingerprint"
    provider._data_fingerprint = "data-fingerprint"
    provider._context_independent_features = True
    provider._ignore_bos_features = True
    provider._batch_segment_size = 4
    provider._batch_segment_mode = "fixed_tokens"
    provider._first_shard_path = None
    provider._batch_tolerance_profile = _profile(max(2, batch_size))
    provider._materialization_prefetch = prefetch
    provider._source_index_loaded = False
    provider._source_index_request = object()  # ty: ignore[invalid-assignment]
    provider._batch_verified = False
    provider._batch_shadow_counter = 0
    provider._batch_validation_observations = []
    provider._batch_validation_anchor_target_keys = None
    provider._batch_pair_timing_sink = None
    provider._allow_unvalidated_batch_profile = True
    provider._batch_runtime_fingerprint_cache = None
    provider._batch_companion_token_strings = {}
    provider._positive_docs = {
        target.target_key: [
            _doc_ref(target.feature_ids[0], index, positive=True) for index in range(positive_docs)
        ]
        for target in targets
    }
    provider._positive_doc_scores = {
        target.target_key: {
            ref.stable_key(): float(positive_docs - index)
            for index, ref in enumerate(provider._positive_docs[target.target_key])
        }
        for target in targets
    }
    provider._intruder_docs = [_doc_ref(0, index, positive=False) for index in range(intruder_docs)]
    provider._model = None
    provider._native_model = object()  # ty: ignore[invalid-assignment]
    provider._sae = object()  # ty: ignore[invalid-assignment]
    provider._tokenizer = SimpleNamespace(
        decode=lambda token_ids: f"token-{token_ids[0]}",
    )
    provider._activation_backend = "native"
    provider._activation_feature_ids = tuple(target.feature_ids[0] for target in targets)
    provider._activation_type = object()  # ty: ignore[invalid-assignment]
    provider._ignore_bos_token_id = 2
    provider._layer = 1
    provider._hook_name = "blocks.1.hook_resid_post"

    executed_blocks: list[int] = []
    prep_threads: list[str] = []

    def _validate_batch_support(self: CachedTokenExampleProvider, *, batch_size: int) -> None:
        assert batch_size == self._cfg.token_examples.materialization_batch_size

    def _ensure_source_index_loaded(self: CachedTokenExampleProvider) -> None:
        self._source_index_loaded = True

    def _ensure_model_loaded(self: CachedTokenExampleProvider) -> None:
        return None

    def _validate_loaded_batch_model(
        self: CachedTokenExampleProvider,
        *,
        batch_size: int,
    ) -> None:
        assert batch_size == self._cfg.token_examples.materialization_batch_size

    def _prepare_doc(
        self: CachedTokenExampleProvider,
        *,
        block,
        slot: int,
        target: ScoringTarget,
        phase: MaterializationPhase,
        doc_ref: SourceDocRef,
    ) -> PreparedDoc:
        prep_threads.append(threading.current_thread().name)
        return PreparedDoc(
            block_ordinal=block.ordinal,
            slot=slot,
            target_key=target.target_key,
            feature_id=target.feature_ids[0],
            phase=phase,
            doc_ref=doc_ref,
            source_text_hash=f"hash-{doc_ref.doc_id}",
            model_token_ids=(2, 10, 11, 12, 13),
            real_token_ids=(10, 11, 12, 13),
            token_strings=("a", "b", "c", "d"),
            token_offset=0,
            activation_slice_start=1,
        )

    def _execute_prepared_batch(
        self: CachedTokenExampleProvider,
        prepared: PreparedBatch,
        *,
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[_DocActivations | None, ...]:
        assert gpu_executor is not None
        executed_blocks.append(prepared.block.block_id)
        docs: list[_DocActivations | None] = []
        for row in prepared.rows:
            if row is None:
                docs.append(None)
                continue
            values = (
                (0.0, float(row.feature_id + row.doc_ref.doc_id % 7 + 1), 0.0, 0.0)
                if row.phase is MaterializationPhase.positive
                else (0.0, 0.0, 0.0, 0.0)
            )
            docs.append(
                _DocActivations(
                    source_text_hash=row.source_text_hash,
                    token_offset=0,
                    token_ids=row.real_token_ids,
                    token_strings=row.token_strings,
                    sae_feature_ids=(row.feature_id,),
                    sae_acts=t.tensor(values, dtype=t.float32).unsqueeze(1),
                )
            )
        return tuple(docs)

    provider._validate_batch_support = MethodType(_validate_batch_support, provider)  # ty: ignore[invalid-assignment]
    provider._ensure_source_index_loaded = MethodType(_ensure_source_index_loaded, provider)  # ty: ignore[invalid-assignment]
    provider._ensure_model_loaded = MethodType(_ensure_model_loaded, provider)  # ty: ignore[invalid-assignment]
    provider._validate_loaded_batch_model = MethodType(_validate_loaded_batch_model, provider)  # ty: ignore[invalid-assignment]
    provider._prepare_doc = MethodType(_prepare_doc, provider)  # ty: ignore[invalid-assignment]
    provider._execute_prepared_batch = MethodType(_execute_prepared_batch, provider)  # ty: ignore[invalid-assignment]
    return provider, executed_blocks, prep_threads


def test_batch_size_one_bulk_entrypoint_delegates_to_unchanged_serial_method(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(
        tmp_path,
        targets,
        batch_size=1,
    )
    calls: list[str] = []

    def _serial(target: ScoringTarget) -> TokenExampleSet:
        calls.append(target.target_key)
        return _sentinel_examples(target, target.feature_ids[0])

    cast(Any, provider).examples_for_target = _serial
    cast(Any, provider)._validate_batch_support = lambda **_kwargs: pytest.fail(
        "batch-size-one must not validate batch support"
    )

    result = provider.examples_for_targets(targets)

    assert calls == ["feature:0", "feature:1"]
    assert result == {
        target.target_key: _sentinel_examples(target, target.feature_ids[0]) for target in targets
    }


def test_public_provider_surfaces_fail_before_prepare(tmp_path: Path) -> None:
    target = _target(0)
    provider, _executed, _threads = _harness_provider(tmp_path, [target])
    provider._prepared = False

    with pytest.raises(AutoInterpDataError, match="provider was not prepared"):
        provider.cache_context()
    with pytest.raises(AutoInterpDataError, match="provider was not prepared"):
        provider.examples_for_target(target)


def test_source_index_double_load_is_an_exact_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    provider._source_index_loaded = True
    monkeypatch.setattr(
        token_examples_module,
        "load_or_build_source_index",
        lambda _request: pytest.fail("an already-loaded source index must not reload"),
    )
    CachedTokenExampleProvider._ensure_source_index_loaded(provider)


def test_positive_doc_topk_rejects_worse_new_rows_without_leaking_scores(
    tmp_path: Path,
) -> None:
    target = _target(0)
    provider, _executed, _threads = _harness_provider(tmp_path, [target])
    provider._cfg = replace(
        provider._cfg,
        token_examples=replace(
            provider._cfg.token_examples,
            max_positive_source_docs_per_target=1,
        ),
    )
    provider._positive_docs[target.target_key] = []
    provider._positive_doc_scores[target.target_key] = {}
    kept = _doc_ref(0, 0, positive=True)
    rejected = _doc_ref(0, 1, positive=True)

    provider._add_positive_doc(target, kept, 10.0)
    provider._add_positive_doc(target, rejected, 1.0)

    assert provider._positive_docs[target.target_key] == [kept]
    assert provider._positive_doc_scores[target.target_key] == {kept.stable_key(): 10.0}


def test_intruder_reservoir_covers_duplicate_replace_and_no_replace_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    provider._cfg = replace(
        provider._cfg,
        token_examples=replace(
            provider._cfg.token_examples,
            max_intruder_source_docs=1,
        ),
    )
    provider._intruder_docs = []
    provider._intruder_doc_keys = set()
    provider._intruder_doc_seen = 0
    first = _doc_ref(0, 0, positive=False)
    replacement = _doc_ref(0, 1, positive=False)
    unsampled = _doc_ref(0, 2, positive=False)

    provider._add_intruder_doc(first)
    provider._add_intruder_doc(first)
    monkeypatch.setattr(
        token_examples_module.random,
        "Random",
        lambda _seed: SimpleNamespace(randrange=lambda _seen: 0),
    )
    provider._add_intruder_doc(replacement)
    monkeypatch.setattr(
        token_examples_module.random,
        "Random",
        lambda _seed: SimpleNamespace(randrange=lambda seen: seen - 1),
    )
    provider._add_intruder_doc(unsampled)

    assert provider._intruder_docs == [replacement]
    assert provider._intruder_doc_seen == 3
    assert provider._intruder_doc_keys == {
        first.stable_key(),
        replacement.stable_key(),
        unsampled.stable_key(),
    }


def test_cached_examples_reject_target_feature_mismatch(tmp_path: Path) -> None:
    expected = _target(0)
    wrong = _sentinel_examples(_target(1), marker=1)
    provider, _executed, _threads = _harness_provider(tmp_path, [expected])

    with pytest.raises(AutoInterpDataError, match="target ids"):
        provider._validate_cached_examples(expected, wrong)


def test_empty_prepared_batch_plan_returns_without_support_or_model_work(
    tmp_path: Path,
) -> None:
    provider, executed, _threads = _harness_provider(tmp_path, [], batch_size=2)
    cast(Any, provider)._validate_batch_support = lambda **_kwargs: pytest.fail(
        "empty plan must not validate a CUDA-only execution path"
    )
    cast(Any, provider)._ensure_model_loaded = lambda: pytest.fail(
        "empty plan must not load the model"
    )

    assert provider.examples_for_targets([]) == {}
    assert executed == []


def test_prepare_success_pins_parsed_fixed_shape_and_source_index_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "shard_00000_features.safetensors.zst"
    parquet = tmp_path / "shard_00000.parquet"
    shard.write_bytes(b"feature-shard")
    parquet.write_bytes(b"parquet")
    cfg = replace(
        _config(tmp_path, batch_size=2),
        data_dir=str(tmp_path),
        source_index=replace(_config(tmp_path, batch_size=2).source_index, enabled=True),
    )
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=32,
        batch_tolerance_profile=_profile(2),
        allow_unvalidated_batch_profile=True,
    )
    requests = []
    monkeypatch.setattr(token_examples_module, "glob_features", lambda _path: [shard])
    monkeypatch.setattr(
        token_examples_module,
        "read_shard_metadata",
        lambda _path: {"segment_size": "4", "segment_mode": "fixed_tokens"},
    )
    monkeypatch.setattr(
        token_examples_module,
        "list_parquet_files",
        lambda _path: [str(parquet)],
    )

    def _validate(request):
        requests.append(request)
        return request

    monkeypatch.setattr(token_examples_module, "validate_source_index_cache", _validate)

    provider.prepare([_target(0), _target(1)], scoring_targets=[_target(1)])

    assert provider._prepared is True
    assert provider._batch_segment_size == 4
    assert provider._batch_segment_mode == "fixed_tokens"
    assert provider._activation_feature_ids == (1,)
    assert len(requests) == 1
    assert requests[0].target_feature_ids == (0, 1)
    assert requests[0].load_target_feature_ids == (1,)


@pytest.mark.parametrize(
    "requested",
    [
        [_target(1), _target(0)],
        [_target(0)],
        [_target(0), _target(1), _target(2)],
        [_target(0), replace(_target(1), display_name="different")],
    ],
)
def test_bulk_request_must_exactly_match_prepared_order_and_values(
    tmp_path: Path,
    requested: list[ScoringTarget],
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    with pytest.raises(AutoInterpDataError, match="exactly match.*ordered target plan"):
        provider.examples_for_targets(requested)


def test_bulk_entrypoint_rejects_unprepared_provider(tmp_path: Path) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    provider._prepared = False
    with pytest.raises(AutoInterpDataError, match="was not prepared"):
        provider.examples_for_targets([_target(0)])


def test_bulk_result_guards_reject_overlap_and_wrong_final_target_set(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(2)]
    provider, _executed, _threads = _harness_provider(tmp_path / "overlap", targets)
    namespace = provider._cache_namespace(targets[0], serial_reference=False)
    cached = _sentinel_examples(targets[0], 10)
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        cache.set(namespace=namespace, target_key=targets[0].target_key, examples=cached)
    cast(Any, provider)._build_and_commit_blocks = lambda *_args, **_kwargs: {
        targets[0].target_key: cached,
        targets[1].target_key: _sentinel_examples(targets[1], 20),
    }
    with pytest.raises(AutoInterpDataError, match="duplicated cache-hit targets"):
        provider.examples_for_targets(targets)

    provider, _executed, _threads = _harness_provider(
        tmp_path / "wrong-set",
        [_target(0)],
    )
    cast(Any, provider)._build_and_commit_blocks = lambda *_args, **_kwargs: {}
    with pytest.raises(AutoInterpDataError, match="returned the wrong target set"):
        provider.examples_for_targets([_target(0)])


def test_partial_block_cache_is_rebuilt_in_full_then_complete_hit_is_lazy(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    provider, executed, _threads = _harness_provider(tmp_path, targets)
    namespace0 = provider._cache_namespace(targets[0], serial_reference=False)
    namespace1 = provider._cache_namespace(targets[1], serial_reference=False)
    stale = _sentinel_examples(targets[0], 999)
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        cache.set(namespace=namespace0, target_key=targets[0].target_key, examples=stale)

    first = provider.examples_for_targets(targets)

    assert executed
    assert first[targets[0].target_key] != stale
    assert provider._source_index_loaded
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        assert (
            cache.get(namespace=namespace0, target_key=targets[0].target_key)
            == first[targets[0].target_key]
        )
        assert (
            cache.get(namespace=namespace1, target_key=targets[1].target_key)
            == first[targets[1].target_key]
        )

    second_provider, second_executed, _ = _harness_provider(tmp_path, targets)
    cast(Any, second_provider)._ensure_source_index_loaded = lambda: pytest.fail(
        "complete block hit must not load source index"
    )
    cast(Any, second_provider)._ensure_model_loaded = lambda: pytest.fail(
        "complete block hit must not load model"
    )
    second = second_provider.examples_for_targets(targets)
    assert second == first
    assert second_executed == []


def test_two_lane_pipeline_alternates_deterministically_and_commits_blocks_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_target(0), _target(1), _target(2), _target(3)]
    provider, executed, prep_threads = _harness_provider(
        tmp_path,
        targets,
        positive_docs=3,
        intruder_docs=4,
    )
    pair_timings: list[dict[str, object]] = []
    provider._batch_pair_timing_sink = pair_timings
    commit_order: list[tuple[str, ...]] = []
    tensor_assembly_threads: list[str] = []
    original = TokenExampleCache.set_many
    original_assemble = PreparedBatch._assemble_token_tensor

    def _record_set_many(self: TokenExampleCache, rows) -> None:
        materialized = list(rows)
        commit_order.append(tuple(target_key for _namespace, target_key, _examples in materialized))
        original(self, materialized)

    monkeypatch.setattr(TokenExampleCache, "set_many", _record_set_many)

    def _record_assemble(self: PreparedBatch) -> t.Tensor:
        tensor_assembly_threads.append(threading.current_thread().name)
        return original_assemble(self)

    monkeypatch.setattr(PreparedBatch, "_assemble_token_tensor", _record_assemble)

    result = provider.examples_for_targets(targets)

    assert set(result) == {target.target_key for target in targets}
    assert executed[:4] == [0, 1, 0, 1]
    assert commit_order == [("feature:0", "feature:1"), ("feature:2", "feature:3")]
    assert prep_threads
    assert set(prep_threads) == {"autointerp-token-prep_0"}
    assert tensor_assembly_threads
    assert set(tensor_assembly_threads) == {"autointerp-token-prep_0"}
    assert len(pair_timings) == 1
    assert pair_timings[0]["block_ids"] == (0, 1)
    assert pair_timings[0]["scope"] == "complete_bulk_partition_v1"
    assert pair_timings[0]["target_keys"] == (
        "feature:0",
        "feature:1",
        "feature:2",
        "feature:3",
    )
    assert cast(int, pair_timings[0]["elapsed_ns"]) > 0


def test_companion_probe_decoding_finishes_on_main_before_prefetch_worker_starts(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, prep_threads = _harness_provider(
        tmp_path,
        targets,
        prefetch=True,
    )
    decode_threads: list[str] = []

    def _decode(token_ids: list[int]) -> str:
        decode_threads.append(threading.current_thread().name)
        return f"token-{token_ids[0]}"

    provider._tokenizer = SimpleNamespace(decode=_decode)

    provider.examples_for_targets(targets)

    assert decode_threads == [threading.current_thread().name] * 2
    assert prep_threads
    assert set(prep_threads) == {"autointerp-token-prep_0"}
    assert provider._batch_companion_token_strings == {
        0: "token-0",
        2: "token-2",
    }


def test_two_lane_scheduler_skips_completed_short_lane_while_long_lane_finishes(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(2)]
    provider, executed, _threads = _harness_provider(
        tmp_path,
        targets,
        prefetch=False,
        positive_docs=3,
        intruder_docs=2,
    )
    provider._positive_docs["feature:2"] = []
    provider._positive_doc_scores["feature:2"] = {}

    result = provider.examples_for_targets(targets)

    assert set(result) == {"feature:0", "feature:2"}
    assert executed[:4] == [0, 1, 0, 1]
    assert executed.count(0) > executed.count(1)
    assert result["feature:0"].positives
    assert result["feature:2"].positives == ()


def test_pair_timing_reconciliation_partitions_complete_bulk_elapsed_exactly(
    tmp_path: Path,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    prefix: dict[str, object] = {"preexisting": True}
    timing_sink: list[dict[str, object]] = [
        prefix,
        {"block_ids": (0, 1), "elapsed_ns": 10},
        {"block_ids": (2,), "elapsed_ns": 20},
    ]
    provider._batch_pair_timing_sink = timing_sink

    provider._reconcile_batch_pair_timings(
        timing_start_index=1,
        bulk_elapsed_ns=100,
        lookup_elapsed_ns_by_block={0: 5, 1: 7, 2: 3},
    )

    assert timing_sink[0] is prefix
    reconciled = timing_sink[1:]
    assert [timing["elapsed_ns"] for timing in reconciled] == [77, 23]
    assert sum(cast(int, timing["elapsed_ns"]) for timing in reconciled) == 100
    assert {timing["scope"] for timing in reconciled} == {"complete_bulk_partition_v1"}
    provider._reconcile_batch_pair_timings(
        timing_start_index=len(timing_sink),
        bulk_elapsed_ns=1,
        lookup_elapsed_ns_by_block={},
    )


@pytest.mark.parametrize(
    ("start_index", "bulk_elapsed_ns", "lookup", "timings", "fragment"),
    [
        (True, 10, {}, [], "start index"),
        (0, 0, {}, [], "bulk elapsed"),
        (0, 10, {True: 1}, [], "lookup map"),
        (0, 10, {0: -1}, [], "lookup map"),
        (0, 10, {0: 1}, [{"block_ids": (), "elapsed_ns": 1}], "timing 0"),
        (0, 10, {}, [{"block_ids": (0,), "elapsed_ns": 1}], "timing 0"),
        (
            0,
            10,
            {0: 5},
            [{"block_ids": (0,), "elapsed_ns": 6}],
            "exceed their enclosing",
        ),
    ],
)
def test_pair_timing_reconciliation_rejects_malformed_or_impossible_inputs(
    tmp_path: Path,
    start_index,
    bulk_elapsed_ns,
    lookup,
    timings,
    fragment: str,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    provider._batch_pair_timing_sink = timings
    domain_field = (
        "timing_start_index"
        if isinstance(start_index, bool)
        else "bulk_elapsed_ns"
        if bulk_elapsed_ns <= 0
        else "lookup_elapsed_ns_by_block"
        if lookup in ({True: 1}, {0: -1})
        else None
    )
    error_type = BeartypeCallHintParamViolation if domain_field else AutoInterpDataError
    with pytest.raises(error_type, match=domain_field or fragment):
        provider._reconcile_batch_pair_timings(
            timing_start_index=start_index,
            bulk_elapsed_ns=bulk_elapsed_ns,
            lookup_elapsed_ns_by_block=lookup,
        )


def test_pair_timing_reconciliation_rechecks_the_final_partition_sum(
    tmp_path: Path,
) -> None:
    class _MutatingTiming(dict):
        elapsed_gets = 0

        def get(self, key, default=None):
            value = super().get(key, default)
            if key == "elapsed_ns":
                self.elapsed_gets += 1
                if self.elapsed_gets == 2:
                    return cast(int, value) + 1
            return value

    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    provider._batch_pair_timing_sink = [
        cast(dict[str, object], _MutatingTiming(block_ids=(0,), elapsed_ns=1))
    ]
    with pytest.raises(AutoInterpDataError, match="do not partition"):
        provider._reconcile_batch_pair_timings(
            timing_start_index=0,
            bulk_elapsed_ns=10,
            lookup_elapsed_ns_by_block={0: 0},
        )


def test_scheduler_rejects_duplicate_future_missing_future_and_malformed_cpu_work(
    tmp_path: Path,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=2,
        targets_by_slot=(_target(0), None),
    )
    future: Future[PreparedBatch] = Future()
    duplicate = _BlockBuild(block=block, states_by_slot=(None, None), future=future)
    with pytest.raises(AutoInterpDataError, match="already has an outstanding"):
        provider._schedule_next_prepared_batch(duplicate, executor=None)

    missing = _BlockBuild(block=block, states_by_slot=(None, None))
    with pytest.raises(AutoInterpDataError, match="no prepared batch"):
        provider._take_prepared_batch(missing)
    with pytest.raises(AutoInterpDataError, match="work row count"):
        provider._prepare_batch(block, (None,))
    with pytest.raises(AutoInterpDataError, match="work for empty slot"):
        provider._prepare_batch(
            block,
            (None, (MaterializationPhase.positive, _doc_ref(1, 0, positive=True))),
        )


def test_synchronous_cpu_prep_exception_is_preserved_in_future(
    tmp_path: Path,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=2,
        targets_by_slot=(_target(0), None),
    )
    build = _BlockBuild(block=block, states_by_slot=(None, None))
    cast(Any, provider)._prepare_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("injected synchronous prep failure")
    )
    provider._schedule_next_prepared_batch(build, executor=None)
    with pytest.raises(RuntimeError, match="injected synchronous prep failure"):
        provider._take_prepared_batch(build)


@pytest.mark.parametrize("failure", ["dummy", "lost"])
def test_pipeline_rejects_executor_dummy_output_or_lost_real_output(
    tmp_path: Path,
    failure: str,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0)])
    original = provider._execute_prepared_batch

    def _invalid(
        prepared: PreparedBatch,
        *,
        gpu_executor: ThreadPoolExecutor | None = None,
    ):
        docs = list(original(prepared, gpu_executor=gpu_executor))
        if failure == "dummy":
            docs[1] = docs[0]
        else:
            docs[0] = None
        return tuple(docs)

    cast(Any, provider)._execute_prepared_batch = _invalid
    message = "document for dummy slot" if failure == "dummy" else "lost real slot"
    with pytest.raises(AutoInterpDataError, match=message):
        provider.examples_for_targets([_target(0)])


def test_prefetch_on_and_off_are_exactly_equal(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1), _target(2), _target(3)]
    on_provider, on_executed, on_threads = _harness_provider(
        tmp_path / "on",
        targets,
        prefetch=True,
    )
    off_provider, off_executed, off_threads = _harness_provider(
        tmp_path / "off",
        targets,
        prefetch=False,
    )

    on = on_provider.examples_for_targets(targets)
    off = off_provider.examples_for_targets(targets)

    assert on == off
    assert on_executed == off_executed
    assert set(on_threads) == {"autointerp-token-prep_0"}
    assert set(off_threads) == {threading.current_thread().name}


def test_cpu_prep_error_cleans_up_worker_and_commits_no_block(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1), _target(2), _target(3)]
    provider, _executed, _threads = _harness_provider(tmp_path, targets, prefetch=True)
    original_prepare = provider._prepare_doc

    def _fail_on_feature_two(**kwargs):
        if kwargs["target"].feature_ids[0] == 2:
            raise RuntimeError("injected CPU prep failure")
        return original_prepare(**kwargs)

    cast(Any, provider)._prepare_doc = _fail_on_feature_two

    with pytest.raises(RuntimeError, match="injected CPU prep failure"):
        provider.examples_for_targets(targets)

    assert not any(
        thread.name.startswith("autointerp-token-prep") for thread in threading.enumerate()
    )
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        for target in targets:
            namespace = provider._cache_namespace(target, serial_reference=False)
            assert cache.get(namespace=namespace, target_key=target.target_key) is None


@pytest.mark.parametrize("inject_failure", [False, True])
def test_gpu_singleton_executor_is_reused_and_shutdown_on_success_or_failure(
    tmp_path: Path,
    inject_failure: bool,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(
        tmp_path,
        targets,
        prefetch=False,
    )
    original_execute = provider._execute_prepared_batch
    executor_ids: list[int] = []
    gpu_thread_names: list[str] = []
    calls = 0

    def _tracked_execute(
        prepared: PreparedBatch,
        *,
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[_DocActivations | None, ...]:
        nonlocal calls
        calls += 1
        assert gpu_executor is not None
        executor_ids.append(id(gpu_executor))
        gpu_thread_names.append(
            gpu_executor.submit(lambda: threading.current_thread().name).result()
        )
        if inject_failure and calls == 2:
            raise RuntimeError("injected persistent GPU executor failure")
        return original_execute(prepared, gpu_executor=gpu_executor)

    cast(Any, provider)._execute_prepared_batch = _tracked_execute
    if inject_failure:
        with pytest.raises(RuntimeError, match="persistent GPU executor failure"):
            provider.examples_for_targets(targets)
    else:
        result = provider.examples_for_targets(targets)
        assert set(result) == {target.target_key for target in targets}

    assert calls >= 2
    assert len(set(executor_ids)) == 1
    assert set(gpu_thread_names) == {"autointerp-token-gpu_0"}
    assert not any(
        thread.name.startswith("autointerp-token-gpu") for thread in threading.enumerate()
    )


def test_executor_failure_before_pair_barrier_leaves_existing_partial_row_untouched(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1), _target(2), _target(3)]
    provider, _executed, _threads = _harness_provider(tmp_path, targets, prefetch=False)
    namespace0 = provider._cache_namespace(targets[0], serial_reference=False)
    stale = _sentinel_examples(targets[0], 777)
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        cache.set(namespace=namespace0, target_key="feature:0", examples=stale)
    original_execute = provider._execute_prepared_batch

    def _fail_second_block(
        prepared: PreparedBatch,
        *,
        gpu_executor: ThreadPoolExecutor | None = None,
    ):
        if prepared.block.block_id == 1:
            raise RuntimeError("injected GPU failure")
        return original_execute(prepared, gpu_executor=gpu_executor)

    cast(Any, provider)._execute_prepared_batch = _fail_second_block

    with pytest.raises(RuntimeError, match="injected GPU failure"):
        provider.examples_for_targets(targets)

    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        assert cache.get(namespace=namespace0, target_key="feature:0") == stale
        for target in targets[1:]:
            namespace = provider._cache_namespace(target, serial_reference=False)
            assert cache.get(namespace=namespace, target_key=target.target_key) is None


def test_final_block_shadow_failure_occurs_before_atomic_commit(tmp_path: Path) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(tmp_path, targets, prefetch=False)
    provider._batch_verified = True

    def _wrong_serial(_row: PreparedDoc, *, capture_residual: bool):
        assert capture_residual is False
        return t.full((4,), 999.0, dtype=t.float32), None

    cast(Any, provider)._serial_surfaces_for_prepared_doc = _wrong_serial

    with pytest.raises(
        AutoInterpBatchValidationError,
        match="numeric validation failed.*final block=0",
    ):
        provider.examples_for_targets(targets)

    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        for target in targets:
            namespace = provider._cache_namespace(target, serial_reference=False)
            assert cache.get(namespace=namespace, target_key=target.target_key) is None


def test_built_block_is_fully_validated_before_atomic_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(
        tmp_path,
        targets,
        prefetch=False,
    )
    writes = 0
    original_set_many = TokenExampleCache.set_many

    def _record_write(self: TokenExampleCache, rows) -> None:
        nonlocal writes
        writes += 1
        original_set_many(self, rows)

    monkeypatch.setattr(TokenExampleCache, "set_many", _record_write)

    def _reject(_target: ScoringTarget, _examples: TokenExampleSet) -> None:
        raise AutoInterpDataError("injected precommit validation failure")

    cast(Any, provider)._validate_cached_examples = _reject

    with pytest.raises(AutoInterpDataError, match="precommit validation failure"):
        provider.examples_for_targets(targets)

    assert writes == 0
    with TokenExampleCache(Path(provider._cfg.token_example_cache_path)) as cache:
        for target in targets:
            assert (
                cache.get(
                    namespace=provider._cache_namespace(target, serial_reference=False),
                    target_key=target.target_key,
                )
                is None
            )


def test_batch_support_error_enumerates_every_unsupported_dimension_and_remediation(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(tmp_path, targets)
    provider._cfg = replace(
        provider._cfg,
        token_examples=replace(
            provider._cfg.token_examples,
            device="cpu",
            activation_backend="auto",
        ),
    )
    provider._context_independent_features = False
    provider._ignore_bos_features = False
    provider._batch_segment_mode = "sentences"
    provider._batch_segment_size = 0

    with pytest.raises(AutoInterpDataError) as caught:
        CachedTokenExampleProvider._validate_batch_support(provider, batch_size=2)
    message = str(caught.value)
    assert "activation_backend='auto'" in message
    assert "device='cpu'" in message
    assert "not context-independent" in message
    assert "do not ignore BOS" in message
    assert "segment_mode='sentences'" in message
    assert "segment_size=0" in message
    assert "materialization_batch_size=1" in message


def test_batch_support_requires_exact_calibrated_model_sae_and_input_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    monkeypatch.setattr(
        token_examples_module,
        "resolve_sae",
        lambda **_kwargs: SimpleNamespace(
            release="test-release",
            sae_id="test-sae",
            hook_name="blocks.1.hook_resid_post",
        ),
    )

    CachedTokenExampleProvider._validate_batch_support(provider, batch_size=2)

    provider._batch_tolerance_profile = replace(
        provider._batch_tolerance_profile,
        calibration_contract=replace(
            provider._batch_tolerance_profile.calibration_contract,
            model_name="gemma-3-different",
        ),
    )
    with pytest.raises(AutoInterpDataError) as caught:
        CachedTokenExampleProvider._validate_batch_support(provider, batch_size=2)
    message = str(caught.value)
    assert "model/SAE/input surface is not covered" in message
    assert "gemma-3-different" in message
    assert "gemma-3-4b-pt" in message
    assert "materialization_batch_size=1" in message


def test_batch_support_rejects_profile_batch_size_before_resolution(
    tmp_path: Path,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    provider._batch_tolerance_profile = _profile(128)
    with pytest.raises(AutoInterpDataError, match="profile batch_size=128.*does not match 2"):
        CachedTokenExampleProvider._validate_batch_support(provider, batch_size=2)


def test_normal_batch_support_rejects_unvalidated_profile_before_resolution(
    tmp_path: Path,
) -> None:
    provider, _executed, _threads = _harness_provider(
        tmp_path,
        [_target(0), _target(1)],
    )
    provider._allow_unvalidated_batch_profile = False

    with pytest.raises(AutoInterpDataError) as caught:
        CachedTokenExampleProvider._validate_batch_support(provider, batch_size=2)

    message = str(caught.value)
    assert "no calibrated runtime fingerprint" in message
    assert "no calibration-observation digest" in message
    assert "materialization_batch_size=1" in message


class _LoadedContractBase(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Identity(), t.nn.Identity()])
        self.norm = t.nn.Identity()
        self.config = SimpleNamespace(
            _attn_implementation="sdpa",
            num_hidden_layers=2,
            layer_types=["full_attention", "sliding_attention"],
            hidden_size=2,
        )
        self.eval()


class _LoadedContractNative(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.zeros(1, dtype=t.bfloat16))
        self.model_name = "gemma-3-4b-pt"
        self.truncated_layer = 1
        self.residual_only = True
        self._use_compiled = False
        self._base = _LoadedContractBase()


class _LoadedContractSAE(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(t.zeros((2, 4), dtype=t.bfloat16))
        self.b_enc = t.nn.Parameter(t.zeros(4, dtype=t.bfloat16))
        self.b_dec = t.nn.Parameter(t.zeros(2, dtype=t.bfloat16))
        self.threshold = t.nn.Parameter(
            t.zeros(4, dtype=t.bfloat16),
            requires_grad=False,
        )


def _loaded_contract_provider(
    tmp_path: Path,
) -> CachedTokenExampleProvider:
    provider, _executed, _threads = _harness_provider(
        tmp_path,
        [_target(0), _target(1)],
    )
    provider._cfg = replace(
        provider._cfg,
        token_examples=replace(provider._cfg.token_examples, device="cpu"),
    )
    cast(Any, provider)._native_model = _LoadedContractNative()
    cast(Any, provider)._sae = _LoadedContractSAE()
    provider._tokenizer = SimpleNamespace(bos_token_id=2)
    provider._activation_backend = "native"
    return provider


def test_loaded_batch_model_accepts_exact_dtype_device_collector_and_sae_contract(
    tmp_path: Path,
) -> None:
    provider = _loaded_contract_provider(tmp_path)
    CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)


def test_loaded_batch_model_reports_all_execution_contract_violations(
    tmp_path: Path,
) -> None:
    provider = _loaded_contract_provider(tmp_path)
    native = cast(_LoadedContractNative, provider._native_model)
    sae = cast(_LoadedContractSAE, provider._sae)
    native.weight = t.nn.Parameter(t.zeros(1, dtype=t.float32))
    native.model_name = "wrong-model"
    native.truncated_layer = 2
    native.residual_only = False
    native._use_compiled = True
    native._base.train()
    native._base.config._attn_implementation = "eager"
    native._base.layers = t.nn.ModuleList([t.nn.Identity()])
    cast(Any, native._base).norm = t.nn.Linear(1, 1)
    native._base.config.num_hidden_layers = 3
    native._base.config.layer_types = ["full_attention"]
    native._base.config.hidden_size = 3
    sae.W_enc = t.nn.Parameter(t.zeros((2, 4), dtype=t.float32))
    sae.b_enc = t.nn.Parameter(t.zeros(3, dtype=t.bfloat16))
    sae.b_dec = t.nn.Parameter(t.zeros(3, dtype=t.bfloat16))
    sae.threshold = t.nn.Parameter(t.zeros(3, dtype=t.bfloat16), requires_grad=False)
    provider._tokenizer = SimpleNamespace(bos_token_id=7)

    with pytest.raises(AutoInterpDataError) as caught:
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)
    message = str(caught.value)
    for fragment in (
        "model dtype",
        "SAE dtype",
        "model_name",
        "truncated layer",
        "not residual-only",
        "compiled forward",
        "not in eval mode",
        "truncated layers",
        "norm is not Identity",
        "num_hidden_layers",
        "layer_types length",
        "hidden_size",
        "attention implementation",
        "BOS id",
        "b_enc shape",
        "b_dec shape",
        "threshold shape",
    ):
        assert fragment in message


def test_loaded_batch_model_rejects_parameterless_or_wrong_device_objects(
    tmp_path: Path,
) -> None:
    provider = _loaded_contract_provider(tmp_path / "parameterless")
    cast(Any, provider._native_model).parameters = lambda: iter(())
    with pytest.raises(AutoInterpDataError, match="native model has no parameters"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)

    provider = _loaded_contract_provider(tmp_path / "device")
    provider._cfg = replace(
        provider._cfg,
        token_examples=replace(provider._cfg.token_examples, device="cuda:7"),
    )
    with pytest.raises(AutoInterpDataError) as caught:
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)
    assert "model device mismatch" in str(caught.value)
    assert "expected=cuda:7" in str(caught.value)
    assert "SAE device mismatch" in str(caught.value)

    provider = _loaded_contract_provider(tmp_path / "layer-types")
    cast(_LoadedContractNative, provider._native_model)._base.config.layer_types = 7
    with pytest.raises(AutoInterpDataError, match="layer_types length=None"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)

    provider = _loaded_contract_provider(tmp_path / "bos-zero")
    provider._tokenizer = SimpleNamespace(bos_token_id=0)
    provider._ignore_bos_token_id = 0
    with pytest.raises(AutoInterpDataError, match="collides with pinned pad/dummy"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)


def test_loaded_batch_model_rejects_missing_dependencies_backend_batch_and_feature_range(
    tmp_path: Path,
) -> None:
    provider = _loaded_contract_provider(tmp_path)
    provider._tokenizer = None
    with pytest.raises(AutoInterpDataError, match="loaded native model, SAE, and tokenizer"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)

    provider = _loaded_contract_provider(tmp_path)
    provider._activation_backend = "transformerlens"
    with pytest.raises(AutoInterpDataError, match="loaded backend"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)

    provider = _loaded_contract_provider(tmp_path)
    with pytest.raises(AutoInterpDataError, match="batch size does not match"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=3)

    provider = _loaded_contract_provider(tmp_path)
    provider._targets = [_target(4)]
    with pytest.raises(AutoInterpDataError, match="out of range.*width 4"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)

    provider = _loaded_contract_provider(tmp_path / "missing-w-enc")
    cast(Any, provider._sae).W_enc = None
    with pytest.raises(AutoInterpDataError, match="W_enc is not a rank-2 tensor"):
        CachedTokenExampleProvider._validate_loaded_batch_model(provider, batch_size=2)


def test_batch_segment_metadata_loader_pins_path_size_and_fixed_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    provider._first_shard_path = tmp_path / "shard.safetensors.zst"
    provider._batch_segment_size = 4
    provider._batch_segment_mode = "fixed_tokens"
    monkeypatch.setattr(
        token_examples_module,
        "read_shard_metadata",
        lambda _path: {"segment_size": "4", "segment_mode": "fixed_tokens"},
    )
    provider._load_batch_segment_metadata_if_needed(force=True)
    assert provider._batch_segment_size == 4
    assert provider._batch_segment_mode == "fixed_tokens"

    monkeypatch.setattr(
        token_examples_module,
        "read_shard_metadata",
        lambda _path: {"segment_size": "8", "segment_mode": "fixed_tokens"},
    )
    with pytest.raises(AutoInterpDataError, match="disagrees.*path=4 shard=8"):
        provider._load_batch_segment_metadata_if_needed(force=True)


@pytest.mark.parametrize(
    ("metadata", "expected_size", "expected_mode"),
    [
        ({"segment_size": "4"}, 4, "fixed_tokens"),
        ({"segment_size": "4", "segment_mode": "sentences"}, 4, "sentences"),
        ({}, 0, "fixed_tokens"),
    ],
)
def test_batch_segment_metadata_loader_handles_missing_and_explicit_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, str],
    expected_size: int,
    expected_mode: str,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    provider._first_shard_path = tmp_path / "shard.safetensors.zst"
    provider._batch_segment_size = 0
    provider._batch_segment_mode = ""
    monkeypatch.setattr(token_examples_module, "read_shard_metadata", lambda _path: metadata)
    provider._load_batch_segment_metadata_if_needed()
    assert provider._batch_segment_size == expected_size
    assert provider._batch_segment_mode == expected_mode


def test_batch_segment_metadata_loader_rejects_missing_shard_and_noninteger_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _executed, _threads = _harness_provider(tmp_path, [_target(0), _target(1)])
    provider._batch_segment_size = 0
    provider._batch_segment_mode = ""
    provider._first_shard_path = None
    with pytest.raises(AutoInterpDataError, match="no feature shard"):
        provider._load_batch_segment_metadata_if_needed()
    provider._first_shard_path = tmp_path / "shard.safetensors.zst"
    monkeypatch.setattr(
        token_examples_module,
        "read_shard_metadata",
        lambda _path: {"segment_size": "not-an-int"},
    )
    with pytest.raises(AutoInterpDataError, match="invalid segment_size"):
        provider._load_batch_segment_metadata_if_needed()


class _BulkProvider:
    def __init__(self, examples: dict[str, TokenExampleSet]) -> None:
        self.examples = examples
        self.bulk_calls: list[list[str]] = []
        self.serial_calls: list[str] = []

    def examples_for_targets(self, targets: list[ScoringTarget]) -> dict[str, TokenExampleSet]:
        self.bulk_calls.append([target.target_key for target in targets])
        return dict(self.examples)

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        self.serial_calls.append(target.target_key)
        return self.examples[target.target_key]


class _SerialProvider:
    def __init__(self, examples: dict[str, TokenExampleSet]) -> None:
        self.examples = examples
        self.calls: list[str] = []

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        self.calls.append(target.target_key)
        return self.examples[target.target_key]


def _sampler_config(tmp_path: Path) -> AutoInterpFilterConfig:
    return replace(
        _config(tmp_path, batch_size=2),
        n_trials_per_target=1,
        n_positive_examples=1,
        n_quantiles=1,
    )


def test_sampler_build_trials_many_uses_bulk_extension_once_and_never_serial(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    examples: dict[str, TokenExampleSet] = {
        str(target.target_key): _sentinel_examples(target, target.feature_ids[0] + 1)
        for target in targets
    }
    provider = _BulkProvider(examples)
    sampler = TokenExampleSampler(
        _sampler_config(tmp_path),
        provider=cast(TokenExampleProvider, provider),
    )

    trials = sampler.build_trials_many(targets)

    assert [len(rows) for rows in trials] == [1, 1]
    assert provider.bulk_calls == [["feature:0", "feature:1"]]
    assert provider.serial_calls == []


def test_sampler_rejects_resolved_blacklist_for_a_different_config_path(
    tmp_path: Path,
) -> None:
    cfg = _sampler_config(tmp_path)
    resolved = replace(
        resolve_document_blacklist(None),
        configured_path="different-blacklist.txt",
    )
    with pytest.raises(AutoInterpConfigError, match="resolved document blacklist path"):
        TokenExampleSampler(
            cfg,
            provider=cast(TokenExampleProvider, _SerialProvider({})),
            document_blacklist=resolved,
        )


def test_sampler_build_trials_many_falls_back_in_exact_target_order(
    tmp_path: Path,
) -> None:
    targets = [_target(1), _target(0)]
    examples: dict[str, TokenExampleSet] = {
        str(target.target_key): _sentinel_examples(target, target.feature_ids[0] + 1)
        for target in targets
    }
    provider = _SerialProvider(examples)
    sampler = TokenExampleSampler(
        _sampler_config(tmp_path),
        provider=cast(TokenExampleProvider, provider),
    )

    trials = sampler.build_trials_many(targets)

    assert [len(rows) for rows in trials] == [1, 1]
    assert provider.calls == ["feature:1", "feature:0"]


@pytest.mark.parametrize(
    ("returned", "fragment"),
    [
        ({"feature:0"}, r"missing=\['feature:1'\]"),
        ({"feature:0", "feature:1", "feature:2"}, r"extra=\['feature:2'\]"),
    ],
)
def test_sampler_rejects_bulk_provider_wrong_target_set(
    tmp_path: Path,
    returned: set[str],
    fragment: str,
) -> None:
    targets = [_target(0), _target(1)]
    all_examples = {
        f"feature:{feature_id}": _sentinel_examples(_target(feature_id), feature_id + 1)
        for feature_id in range(3)
    }
    provider = _BulkProvider({key: all_examples[key] for key in returned})
    sampler = TokenExampleSampler(
        _sampler_config(tmp_path),
        provider=cast(TokenExampleProvider, provider),
    )
    with pytest.raises(AutoInterpDataError, match=fragment):
        sampler.build_trials_many(targets)


def test_sampler_rejects_duplicate_target_plan_before_provider_side_effects(
    tmp_path: Path,
) -> None:
    target = _target(0)
    examples: dict[str, TokenExampleSet] = {str(target.target_key): _sentinel_examples(target, 1)}
    bulk_provider = _BulkProvider(examples)
    bulk_sampler = TokenExampleSampler(
        _sampler_config(tmp_path / "bulk"),
        provider=cast(TokenExampleProvider, bulk_provider),
    )
    serial_provider = _SerialProvider(examples)
    serial_sampler = TokenExampleSampler(
        _sampler_config(tmp_path / "serial"),
        provider=cast(TokenExampleProvider, serial_provider),
    )

    for sampler in (bulk_sampler, serial_sampler):
        with pytest.raises(AutoInterpDataError, match="duplicate targets"):
            sampler.build_trials_many([target, target])

    assert bulk_provider.bulk_calls == []
    assert bulk_provider.serial_calls == []
    assert serial_provider.calls == []


def test_serial_cache_identity_is_legacy_even_when_config_default_requests_batching(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]
    batched, _executed, _threads = _harness_provider(tmp_path / "shared", targets, batch_size=2)
    serial, _executed2, _threads2 = _harness_provider(tmp_path / "shared", targets, batch_size=1)

    assert batched._cache_namespace(targets[0], serial_reference=True) == serial._cache_namespace(
        targets[0],
        serial_reference=True,
    )
    assert batched._cache_namespace(
        targets[0],
        serial_reference=False,
    ) != serial._cache_namespace(targets[0], serial_reference=True)
    assert serial.cache_context()["method"] == "token_examples_v6"
    assert "materialization" not in serial.cache_context()


def test_batch_cache_identity_pins_plan_shape_policy_profile_runtime_and_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [_target(0), _target(1)]
    provider, _executed, _threads = _harness_provider(tmp_path, targets)
    calls = 0

    def _fingerprint(_device=None):
        nonlocal calls
        calls += 1
        return {"snapshot": "fixed"}

    monkeypatch.setattr(token_examples_module, "runtime_numeric_fingerprint", _fingerprint)
    context = provider.cache_context()
    materialization = context["materialization"]
    assert context["method"] == "token_examples_v7_ci_block_two_stream_singleton_v1"
    assert materialization["batch_size"] == 2
    assert materialization["gpu_singleton_concurrency"] == 2
    assert materialization["model_input_policy"] == "true_length_singleton_v1"
    assert "fixed_model_width" not in materialization
    assert "pad_and_dummy_token_id" not in materialization
    assert materialization["target_plan_sha256"] == target_plan_hash(targets)
    assert materialization["companion_leakage_probe"] == "hold_probe_toggle_other_singleton_v1"
    assert materialization["validation_profile"] == _profile(2).identity()
    assert "runtime_numeric_fingerprint" in materialization
    assert materialization["max_positive_windows_per_target"] == 3
    assert materialization["max_non_activating_windows_per_target"] == 3
    for target in targets:
        provider._cache_namespace(target, serial_reference=False)
    assert provider.cache_context()["materialization"]["runtime_numeric_fingerprint"] == {
        "snapshot": "fixed"
    }
    assert calls == 1

    mutable_context = provider.cache_context()
    mutable_context["materialization"]["runtime_numeric_fingerprint"]["snapshot"] = "mutated"
    assert provider.cache_context()["materialization"]["runtime_numeric_fingerprint"] == {
        "snapshot": "fixed"
    }

    no_prefetch, _executed2, _threads2 = _harness_provider(
        tmp_path,
        targets,
        prefetch=False,
    )
    assert no_prefetch.cache_context() == context

    reversed_plan, _executed3, _threads3 = _harness_provider(
        tmp_path,
        list(reversed(targets)),
    )
    assert reversed_plan.cache_context() != context

    serial, _executed4, _threads4 = _harness_provider(
        tmp_path,
        targets,
        batch_size=1,
    )
    serial.cache_context()
    assert calls == 3  # no_prefetch + reversed each snapshot once; serial never does


def test_runner_manifest_omits_batch_field_for_serial_and_fallback_but_records_real_bulk(
    tmp_path: Path,
) -> None:
    serial_cfg = _config(tmp_path / "serial", batch_size=1)
    batch_cfg = _config(tmp_path / "batch", batch_size=128)
    assert "materialization_batch_size" not in runner_manifest_token_examples(serial_cfg)
    assert "materialization_batch_size" not in runner_manifest_token_examples(
        batch_cfg,
        bulk_materialization=False,
    )
    assert (
        runner_manifest_token_examples(
            batch_cfg,
            bulk_materialization=True,
        )["materialization_batch_size"]
        == 128
    )


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, "128", None])
def test_materialization_batch_size_config_is_strict_positive_int(
    tmp_path: Path,
    bad: Any,
) -> None:
    cfg = _config(tmp_path, batch_size=2)
    with pytest.raises(BeartypeCallHintParamViolation, match="materialization_batch_size"):
        replace(cfg.token_examples, materialization_batch_size=bad)


def test_materialization_batch_size_one_and_default_128_validate(tmp_path: Path) -> None:
    for batch_size in (1, 128):
        cfg = _config(tmp_path / str(batch_size), batch_size=batch_size)
        validate_config(cfg)


def test_runner_completes_one_bulk_materialization_before_any_judge_call(
    tmp_path: Path,
) -> None:
    targets = [_target(0), _target(1)]

    class _RunnerBulkProvider:
        def __init__(self) -> None:
            self.prepared: list[ScoringTarget] = []
            self.bulk_complete = False
            self.bulk_calls = 0
            self.serial_calls = 0

        def prepare(self, prepared: list[ScoringTarget]) -> None:
            self.prepared = list(prepared)

        def cache_context(self) -> dict:
            return {"provider": "runner-bulk-order-v1"}

        def examples_for_targets(
            self,
            requested: list[ScoringTarget],
        ) -> dict[str, TokenExampleSet]:
            assert requested == self.prepared
            self.bulk_calls += 1
            result: dict[str, TokenExampleSet] = {
                str(target.target_key): _sentinel_examples(
                    target,
                    target.feature_ids[0] + 1,
                )
                for target in requested
            }
            self.bulk_complete = True
            return result

        def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
            self.serial_calls += 1
            raise AssertionError(f"runner unexpectedly materialized {target.target_key} serially")

    provider = _RunnerBulkProvider()

    class _OrderingJudge:
        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, trial: IntruderTrial, *, renderer) -> IntruderTrialResult:
            assert provider.bulk_complete
            self.calls += 1
            return IntruderTrialResult(
                trial_key=trial.trial_key,
                prediction=trial.intruder_index,
                correct_index=trial.intruder_index,
                correct=True,
                response=f"[RESPONSE]: {trial.intruder_index}",
                cached=False,
            )

    judge = _OrderingJudge()
    cfg = replace(
        _config(tmp_path, batch_size=128),
        feature_ids=[0, 1],
        n_trials_per_target=2,
        min_trials_for_score=1,
        n_positive_examples=1,
        n_quantiles=1,
        score_threshold=0.0,
        top_k=None,
        fail_on_empty=True,
    )

    report = AutoInterpFilterRunner(
        cfg,
        judge=cast(IntruderDetectionJudge, judge),
        token_example_provider=cast(TokenExampleProvider, provider),
    ).run()

    assert provider.prepared == targets
    assert provider.bulk_calls == 1
    assert provider.serial_calls == 0
    assert judge.calls == 4
    assert report.n_targets_total == 2
    assert report.n_targets_scored == 2
