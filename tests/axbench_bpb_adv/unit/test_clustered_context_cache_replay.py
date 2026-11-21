"""CPU replay, clustered selection, and selected-prefix contracts for G3b."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger
from safetensors.torch import save_file

import steering.conditional.cluster_segment_selection as selection_module
import steering.eval.axbench_bpb_adv.clustered_context_cache as cache_module
from errors import FeatureClustersError
from steering.conditional.delta import DeltaMode, DeltaSamplingWeights
from steering.conditional.feature_clusters import (
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    build_cluster_feature_contract,
)
from steering.eval.context_cache_io import CLUSTER_SELECTION_SEMANTICS
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.stats_types import PerFeatureStat

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _contract(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    definition = FeatureClusterDefinition(clusters=((1, 2), (4, 5)))
    clusters = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=8,
        ),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    return build_cluster_feature_contract(
        feature_mask=mask,
        feature_clusters=clusters,
        statistics=FeatureStatisticsIdentity(
            file_sha256=_SHA_A,
            percentile_table_checksum=_SHA_B,
            n_sae_features=8,
            feature_id_domain=(1, 2, 3, 4, 5, 6),
        ),
        feature_directory=FeatureDirectoryIdentity(
            segment_size=4,
            layer=8,
            activation_type="residual",
            pooling="max_pool",
            width="16k",
            model="gemma-2-2b",
            dataset_phase="base",
            context_independent_features=False,
            ignore_bos_features=False,
        ),
        sae=SaeIdentity(
            release="release",
            sae_id="sae-id",
            hook_name="blocks.8.hook_resid_post",
            model="gemma-2-2b",
            layer=8,
            activation_type="residual",
            width="16k",
            actual_loaded_width=8,
        ),
        checkpoint=CheckpointPin(
            checkpoint_dir=checkpoint_dir,
            resolved_tag="clustered",
            step=7,
            model_file=StableFileDigest(
                path=checkpoint_dir / "model_000007.pt",
                sha256=_SHA_C,
                byte_length=1,
                stat_signature=(1, 2, 1, 3, 4, 5),
            ),
            metadata_file=StableFileDigest(
                path=checkpoint_dir / "meta_000007.json",
                sha256=_SHA_D,
                byte_length=1,
                stat_signature=(1, 3, 1, 3, 4, 5),
            ),
            n_sae_features=8,
            feature_vocab=FeatureVocab.from_mask(mask, 8),
            load_bearing_metadata_json="{}",
        ),
        selection=CheckpointSelectionContract(
            positive_selection_json=json.dumps(
                {
                    "threshold_type": "feature_percentile",
                    "threshold": 0.5,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            negative_selection_json=json.dumps(
                {
                    "distribution": "normal",
                    "mean": 1.0,
                    "std": 0.01,
                    "min_k": 1,
                    "max_k": 2,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            num_buckets=2,
            effective_positive_buckets=1,
            preselection_seed=19,
            selection_hash=_SHA_A,
            selection_manifest_sha256=_SHA_B,
            selection_shards_sha256=_SHA_C,
            packed_row_fingerprint=_SHA_D,
        ),
        concept_source_mode="statistics_domain",
        concept_source_domain=(1, 2, 3, 4, 5, 6),
    )


def _percentile_table() -> t.Tensor:
    return t.arange(1, 100, dtype=t.float64).unsqueeze(0).repeat(8, 1)


def _negative_selection() -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.normal,
        mean=1.0,
        std=0.01,
        min_k=1,
        max_k=2,
    )


def _threshold_replay_state(contract, *, seed: int = 19):
    rng = t.Generator().manual_seed(seed)
    return cache_module.ClusteredReplayState(
        pipeline=FeaturePipelineConfig(
            selection_config=ThresholdSelectionConfig(
                selection=ThresholdSelection(
                    ThresholdType.feature_percentile,
                    0.5,
                ),
                precomputed_thresholds=t.zeros(8, dtype=t.float32),
            ),
            boundaries=t.zeros(8, 0, dtype=t.float64),
            negative_bundle=NegativeBundle(
                selection=_negative_selection(),
                n_sae_features=8,
                rng=rng,
                feature_mask=contract.feature_mask,
            ),
            effective_buckets=1,
            feature_mask=contract.feature_mask,
        ),
        percentile_table=_percentile_table(),
    )


def _sampling_replay_state(contract, *, seed: int = 19):
    rng = t.Generator().manual_seed(seed)
    return cache_module.ClusteredReplayState(
        pipeline=FeaturePipelineConfig(
            selection_config=SamplingBundle(
                selection=SamplingSelection(
                    distribution=Distribution.normal,
                    mean=1.0,
                    std=0.01,
                    min_k=1,
                    max_k=2,
                    mode=SamplingMode.weighted,
                ),
                percentile_table=_percentile_table(),
                rng=rng,
            ),
            boundaries=t.zeros(8, 0, dtype=t.float64),
            negative_bundle=NegativeBundle(
                selection=_negative_selection(),
                n_sae_features=8,
                rng=rng,
                feature_mask=contract.feature_mask,
            ),
            effective_buckets=1,
            feature_mask=contract.feature_mask,
        ),
        percentile_table=_percentile_table(),
    )


def _raw_segment(
    ids: list[int] | None = None,
    magnitudes: list[float] | None = None,
):
    raw_ids = [1, 2, 3] if ids is None else ids
    raw_magnitudes = [20.0, 80.0, 60.0] if magnitudes is None else magnitudes
    return cache_module.RawClusteredFeatureSegment(
        feature_ids=t.tensor(raw_ids, dtype=t.int64),
        feature_magnitudes=t.tensor(raw_magnitudes, dtype=t.float32),
        char_start=0,
        char_end=4,
    )


def _per_feature_stats() -> dict[str, PerFeatureStat]:
    return {
        str(feature_id): PerFeatureStat(
            count=10,
            pct_segments_active=0.5,
            mean_mag=50.0,
            mean_mag_incl_zeros=25.0,
            std_mag=10.0,
            min_mag=1.0,
            max_mag=99.0,
            median_mag=50.0,
            percentiles={
                f"p{percentile}": float(percentile)
                for percentile in range(1, 100)
            },
        )
        for feature_id in range(8)
    }


def _with_statistics_path(contract, path: str, *, sha256: str):
    provenance = contract.provenance_dict()
    provenance["feature_statistics_path"] = path
    return replace(
        contract,
        statistics=replace(contract.statistics, file_sha256=sha256),
        provenance_json=json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _valid_materialized_segments() -> tuple[cache_module._MaterializedSegment, ...]:
    return (
        cache_module._MaterializedSegment(
            context=ContextFeatureSegment(
                feature_ids=(1, 4),
                buckets=(0, 1),
                char_start=0,
                char_end=4,
            ),
            positive_percentiles_uint8=t.tensor([204], dtype=t.uint8),
            n_positive=1,
            n_negative=1,
        ),
        cache_module._MaterializedSegment(
            context=ContextFeatureSegment(
                feature_ids=(),
                buckets=(),
                char_start=4,
                char_end=4,
            ),
            positive_percentiles_uint8=t.empty(0, dtype=t.uint8),
            n_positive=0,
            n_negative=0,
        ),
    )


def _selected_metadata(contract) -> dict[str, str]:
    return cache_module._selected_metadata(
        contract=contract,
        prefix_hash="1" * 16,
        prefix_text_sha256=_SHA_A,
        extraction_identity_sha256=_SHA_B,
        raw_sha256=_SHA_C,
        generator_state_after_sha256=_SHA_D,
    )


def _valid_selected_tensors() -> dict[str, t.Tensor]:
    return {
        "n_segments": t.tensor([2], dtype=t.int64),
        "seg_0_feature_ids": t.tensor([1, 4], dtype=t.int64),
        "seg_0_buckets": t.tensor([0, 1], dtype=t.int64),
        "seg_0_char_bounds": t.tensor([0, 4], dtype=t.int64),
        "seg_0_positive_percentiles_uint8": t.tensor([204], dtype=t.uint8),
        "seg_0_n_positive": t.tensor([1], dtype=t.int64),
        "seg_1_feature_ids": t.empty(0, dtype=t.int64),
        "seg_1_buckets": t.empty(0, dtype=t.int64),
        "seg_1_char_bounds": t.tensor([4, 4], dtype=t.int64),
        "seg_1_positive_percentiles_uint8": t.empty(0, dtype=t.uint8),
        "seg_1_n_positive": t.tensor([0], dtype=t.int64),
    }


def test_replay_constants_and_cache_semantics_are_unchanged() -> None:
    assert cache_module.SELECTED_PREFIX_SCHEMA_VERSION == 1
    assert CLUSTER_SELECTION_SEMANTICS == (
        "pre_mask_representative_max_percentile_v2"
    )


def test_reconstruct_fused_load_parses_the_exact_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"one-verified-statistics-snapshot"
    contract = _with_statistics_path(
        _contract(tmp_path),
        "virtual-statistics.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    fake_stats = SimpleNamespace(
        metadata=SimpleNamespace(n_sae_features=8),
        per_feature_stats=_per_feature_stats(),
    )
    parsed_payloads: list[bytes] = []
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: payload)

    def parse(snapshot: bytes):
        parsed_payloads.append(snapshot)
        return fake_stats, _SHA_B, None

    monkeypatch.setattr(
        cache_module,
        "load_statistics_and_checksum_from_bytes",
        parse,
    )
    monkeypatch.setattr(
        cache_module,
        "compute_percentile_checksum",
        lambda _table: _SHA_B,
    )

    state = cache_module.reconstruct_clustered_replay_state(contract)

    assert parsed_payloads == [payload]
    assert state.percentile_table.shape == (8, 99)
    assert isinstance(state.pipeline.selection_config, ThresholdSelectionConfig)
    assert state.pipeline.negative_bundle is not None
    assert t.equal(
        state.pipeline.negative_bundle.rng.get_state(),
        t.Generator().manual_seed(19).get_state(),
    )


def test_reconstruct_fused_load_rejects_transient_a_to_b_to_a_swap_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content_a = b"stable-content-A"
    content_b = b"transient-content-B"
    contract = _with_statistics_path(
        _contract(tmp_path),
        "virtual-statistics.safetensors",
        sha256=hashlib.sha256(content_a).hexdigest(),
    )
    parse_calls: list[bytes] = []
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: content_b)
    monkeypatch.setattr(
        cache_module,
        "load_statistics_and_checksum_from_bytes",
        lambda payload: (parse_calls.append(payload), (None, None, None))[1],
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="bytes changed while loading clustered selection replay",
    ):
        cache_module.reconstruct_clustered_replay_state(contract)

    assert parse_calls == []


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_path", "missing feature_statistics_path"),
        ("resolve", "could not revalidate feature statistics"),
        ("parse", "could not rebuild percentile table"),
        ("width", "statistics width changed"),
        ("checksum", "checksum does not match"),
        ("json", "selection JSON is malformed"),
        ("json_domain", "must be JSON objects"),
        ("pipeline_build", "could not reconstruct checkpoint feature pipeline"),
        ("mask", "pipeline mask does not match"),
        ("buckets", "effective bucket count does not match"),
        ("private_rng", "missing negative selection RNG"),
    ],
)
def test_reconstruct_validation_failure_map(
    case: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"validated-statistics"
    contract = _with_statistics_path(
        _contract(tmp_path),
        "virtual-statistics.safetensors",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    fake_stats = SimpleNamespace(
        metadata=SimpleNamespace(n_sae_features=8),
        per_feature_stats=_per_feature_stats(),
    )
    pipeline = _threshold_replay_state(contract).pipeline
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: payload)
    monkeypatch.setattr(
        cache_module,
        "load_statistics_and_checksum_from_bytes",
        lambda _payload: (fake_stats, _SHA_B, None),
    )
    monkeypatch.setattr(
        cache_module,
        "compute_percentile_checksum",
        lambda _table: _SHA_B,
    )
    monkeypatch.setattr(
        cache_module,
        "build_feature_pipeline_from_cfg_block",
        lambda *_args, **_kwargs: pipeline,
    )

    if case == "missing_path":
        contract = replace(contract, provenance_json="{}")
    elif case == "resolve":
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda _self, strict=False: (_ for _ in ()).throw(
                FileNotFoundError("missing")
            ),
        )
    elif case == "parse":
        monkeypatch.setattr(
            cache_module,
            "load_statistics_and_checksum_from_bytes",
            lambda _payload: (_ for _ in ()).throw(ValueError("bad snapshot")),
        )
    elif case == "width":
        fake_stats.metadata.n_sae_features = 7
    elif case == "checksum":
        monkeypatch.setattr(
            cache_module,
            "compute_percentile_checksum",
            lambda _table: _SHA_C,
        )
    elif case == "json":
        contract = replace(
            contract,
            selection=replace(
                contract.selection,
                positive_selection_json="{",
            ),
        )
    elif case == "json_domain":
        contract = replace(
            contract,
            selection=replace(
                contract.selection,
                positive_selection_json="[]",
            ),
        )
    elif case == "pipeline_build":
        monkeypatch.setattr(
            cache_module,
            "build_feature_pipeline_from_cfg_block",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("bad pipeline")
            ),
        )
    elif case == "mask":
        pipeline.feature_mask = None
    elif case == "buckets":
        pipeline.effective_buckets = 2
    elif case == "private_rng":
        pipeline.negative_bundle = None
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module.reconstruct_clustered_replay_state(contract)


def test_private_pipeline_generator_returns_without_advancing_callers_rng(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _sampling_replay_state(contract)
    selection = state.pipeline.selection_config
    assert isinstance(selection, SamplingBundle)
    caller_rng = selection.rng
    before = caller_rng.get_state().clone()

    returned = cache_module._private_pipeline_generator(state.pipeline)

    assert returned is caller_rng
    assert t.equal(caller_rng.get_state(), before)
    assert cache_module._generator_state_sha256(state.pipeline) == hashlib.sha256(
        before.numpy().tobytes()
    ).hexdigest()
    assert t.equal(caller_rng.get_state(), before)


def test_private_pipeline_generator_rejects_split_sampling_rng(
    tmp_path: Path,
) -> None:
    state = _sampling_replay_state(_contract(tmp_path))
    assert state.pipeline.negative_bundle is not None
    state.pipeline.negative_bundle.rng = t.Generator().manual_seed(99)

    with pytest.raises(AxBenchBpbAdvError, match="shared positive/negative RNG"):
        cache_module._private_pipeline_generator(state.pipeline)


def test_private_pipeline_generator_rejects_unknown_pipeline_type(
    tmp_path: Path,
) -> None:
    pipeline = _threshold_replay_state(_contract(tmp_path)).pipeline
    pipeline.selection_config = object()

    with pytest.raises(AxBenchBpbAdvError, match="unsupported replay pipeline type"):
        cache_module._private_pipeline_generator(pipeline)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("non_cpu", "resident on CPU"),
        ("non_1d", "one-dimensional"),
        ("misaligned", "misaligned"),
        ("ids_dtype", "IDs must use int64"),
        ("magnitudes_dtype", "magnitudes must use float32"),
        ("nan", "non-finite"),
        ("inf", "non-finite"),
        ("zero", "non-positive"),
        ("negative_magnitude", "non-positive"),
        ("negative_id", "out-of-range"),
        ("too_large_id", "out-of-range"),
        ("duplicate_id", "duplicate"),
    ],
)
def test_select_raw_clustered_segment_rejects_every_invalid_raw_domain(
    case: str,
    match: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _threshold_replay_state(contract)
    ids = t.tensor([1, 3], dtype=t.int64)
    magnitudes = t.tensor([20.0, 60.0], dtype=t.float32)
    if case == "non_cpu":
        ids = t.empty(2, dtype=t.int64, device="meta")
    elif case == "non_1d":
        ids = ids.reshape(1, 2)
        magnitudes = magnitudes.reshape(1, 2)
    elif case == "misaligned":
        magnitudes = magnitudes[:1]
    elif case == "ids_dtype":
        ids = ids.to(t.int32)
    elif case == "magnitudes_dtype":
        magnitudes = magnitudes.to(t.float64)
    elif case == "nan":
        magnitudes[0] = float("nan")
    elif case == "inf":
        magnitudes[0] = float("inf")
    elif case == "zero":
        magnitudes[0] = 0.0
    elif case == "negative_magnitude":
        magnitudes[0] = -1.0
    elif case == "negative_id":
        ids[0] = -1
    elif case == "too_large_id":
        ids[0] = 8
    elif case == "duplicate_id":
        ids[1] = ids[0]
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)
    raw = cache_module.RawClusteredFeatureSegment(
        feature_ids=ids,
        feature_magnitudes=magnitudes,
        char_start=0,
        char_end=4,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module.select_raw_clustered_segment(
            raw,
            replay_state=state,
            contract=contract,
        )


def test_g3b_masked_stat_less_feature_is_logged_and_dropped(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _threshold_replay_state(contract)
    state.percentile_table[0].zero_()
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        result = cache_module.select_raw_clustered_segment(
            _raw_segment(ids=[0, 1], magnitudes=[30.0, 80.0]),
            replay_state=state,
            contract=contract,
        )
    finally:
        logger.remove(sink)

    assert result.positive_ids.tolist() == [1]
    assert 0 not in result.positive_ids.tolist()
    assert any(
        "masked feature(s) [0]" in message
        and "no percentile statistics" in message
        for message in messages
    )


def test_g3b_masked_stat_less_green_test_is_non_vacuous_against_oracle_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen no-predrop oracle chain raises on the green fixture."""

    contract = _contract(tmp_path)
    state = _threshold_replay_state(contract)
    state.percentile_table[0].zero_()

    def frozen_oracle_chain(
        feature_ids: t.Tensor,
        feature_magnitudes: t.Tensor,
        *,
        percentile_table: t.Tensor,
        clusters,
        pipeline,
        source: str,
    ):
        del source
        percentile_ranks = selection_module.magnitude_to_percentile(
            feature_ids,
            feature_magnitudes,
            percentile_table,
        )
        return selection_module.select_clustered_segment(
            feature_ids,
            selection_module.percentile_to_uint8(percentile_ranks),
            clusters=clusters,
            pipeline=pipeline,
        )

    monkeypatch.setattr(
        cache_module,
        "select_clustered_segment_from_magnitudes",
        frozen_oracle_chain,
    )

    with pytest.raises(ValueError, match="all-zero percentile rows"):
        cache_module.select_raw_clustered_segment(
            _raw_segment(ids=[0, 1], magnitudes=[30.0, 80.0]),
            replay_state=state,
            contract=contract,
        )


def test_g3b_unmasked_stat_less_feature_still_raises(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _threshold_replay_state(contract)
    state.percentile_table[1].zero_()

    with pytest.raises(ValueError, match="all-zero percentile rows"):
        cache_module.select_raw_clustered_segment(
            _raw_segment(ids=[1], magnitudes=[80.0]),
            replay_state=state,
            contract=contract,
        )


def test_threshold_and_sampling_replay_branches_are_both_live(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    raw = _raw_segment()
    threshold = cache_module.select_raw_clustered_segment(
        raw,
        replay_state=_threshold_replay_state(contract),
        contract=contract,
    )
    sampling = cache_module.select_raw_clustered_segment(
        raw,
        replay_state=_sampling_replay_state(contract),
        contract=contract,
    )

    assert threshold.positive_ids.tolist() == [1, 3]
    assert threshold.positive_percentiles_uint8.tolist() == [204, 153]
    assert len(threshold.negative_ids) == 1
    assert len(sampling.positive_ids) == 1
    assert len(sampling.negative_ids) == 1


def test_positive_sampling_precedes_negative_sampling_in_g3b_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    state = _sampling_replay_state(contract)
    calls: list[str] = []

    def positive(**_kwargs: Any):
        calls.append("positive")
        return [(t.tensor([1], dtype=t.int64), t.tensor([0.8]))]

    def negative(*_args: Any, **_kwargs: Any):
        calls.append("negative")
        return [t.tensor([4], dtype=t.int64)]

    monkeypatch.setattr(selection_module, "sample_features_batched", positive)
    monkeypatch.setattr(
        selection_module,
        "sample_inactive_features_batched",
        negative,
    )

    materialized = cache_module._materialize_segment(
        _raw_segment(),
        replay_state=state,
        contract=contract,
    )

    assert calls == ["positive", "negative"]
    assert materialized.context.feature_ids == (1, 4)
    assert materialized.context.buckets == (0, 1)
    assert materialized.n_positive == 1
    assert materialized.n_negative == 1


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("non_representative", FeatureClustersError),
        ("masked", FeatureClustersError),
        ("duplicate", AxBenchBpbAdvError),
        ("bucket_range", AxBenchBpbAdvError),
        ("positive_bucket", AxBenchBpbAdvError),
        # Reaches the reserved-negative-bucket check, which is otherwise
        # unfalsifiable: the negative suffix is built by t.full(..., negative_
        # bucket) two statements earlier, so it can only disagree when the
        # positive channels are themselves length-mismatched.
        ("positive_length_mismatch", AxBenchBpbAdvError),
    ],
)
def test_materialize_segment_defense_in_depth_failure_map(
    case: str,
    expected_error: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(tmp_path)
    state = _threshold_replay_state(contract)
    positive_ids = t.tensor([1], dtype=t.int64)
    positive_buckets = t.tensor([0], dtype=t.int64)
    negative_ids = t.tensor([4], dtype=t.int64)
    if case == "non_representative":
        positive_ids = t.tensor([2], dtype=t.int64)
    elif case == "masked":
        positive_ids = t.tensor([0], dtype=t.int64)
    elif case == "duplicate":
        negative_ids = t.tensor([1], dtype=t.int64)
    elif case == "bucket_range":
        positive_buckets = t.tensor([2], dtype=t.int64)
    elif case == "positive_bucket":
        positive_buckets = t.tensor([1], dtype=t.int64)
    elif case == "positive_length_mismatch":
        positive_buckets = t.tensor([0, 0], dtype=t.int64)
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)
    percentiles = (
        t.tensor([204, 204], dtype=t.uint8)
        if case == "positive_length_mismatch"
        else t.tensor([204], dtype=t.uint8)
    )
    malformed = selection_module.ClusteredSegmentSelection(
        positive_ids=positive_ids,
        positive_buckets=positive_buckets,
        positive_percentiles_uint8=percentiles,
        negative_ids=negative_ids,
    )
    monkeypatch.setattr(
        cache_module,
        "select_raw_clustered_segment",
        lambda *_args, **_kwargs: malformed,
    )

    with pytest.raises(expected_error):
        cache_module._materialize_segment(
            _raw_segment(),
            replay_state=state,
            contract=contract,
        )


def test_empty_segment_consumes_zero_draws_and_selects_nothing(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _sampling_replay_state(contract)
    rng = cache_module._private_pipeline_generator(state.pipeline)
    before = rng.get_state().clone()
    empty = cache_module.RawClusteredFeatureSegment(
        feature_ids=t.empty(0, dtype=t.int64),
        feature_magnitudes=t.empty(0, dtype=t.float32),
        char_start=0,
        char_end=0,
    )

    result = cache_module.select_raw_clustered_segment(
        empty,
        replay_state=state,
        contract=contract,
    )

    assert result.positive_ids.numel() == 0
    assert result.negative_ids.numel() == 0
    assert t.equal(rng.get_state(), before)


def test_fixed_seed_replay_golden_selection_and_final_rng(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    state = _sampling_replay_state(contract, seed=20260726)

    materialized = cache_module._materialize_segment(
        _raw_segment(),
        replay_state=state,
        contract=contract,
    )

    assert materialized.context.feature_ids == (3, 6)
    assert materialized.context.buckets == (0, 1)
    assert materialized.positive_percentiles_uint8.tolist() == [153]
    assert cache_module._generator_state_sha256(state.pipeline) == (
        "8fe60a8d16978513cc0bfb4f6f3e859dbf7a9e69f6c60da2a833739ed3212a12"
    )


def test_selected_prefix_round_trip_is_exact_and_reports_totals(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    segments = _valid_materialized_segments()
    metadata = _selected_metadata(contract)

    payload = cache_module._serialize_selected_prefix(
        segments,
        metadata=metadata,
        contract=contract,
    )
    assert payload == cache_module._serialize_selected_prefix(
        segments,
        metadata=metadata,
        contract=contract,
    )
    path = tmp_path / "selected.safetensors"
    path.write_bytes(payload)

    assert cache_module._validate_selected_file(
        path,
        expected_bytes=payload,
        expected_metadata=metadata,
        contract=contract,
    ) == {
        "selected_sha256": hashlib.sha256(payload).hexdigest(),
        "n_segments": 2,
        "n_features_total": 2,
        "n_positive_total": 1,
        "n_negative_total": 1,
    }
    with pytest.raises(AxBenchBpbAdvError, match="byte mismatch"):
        cache_module._validate_selected_file(
            path,
            expected_bytes=payload + b"different",
            expected_metadata=metadata,
            contract=contract,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing", "artifact is missing"),
        ("corrupt", "could not read safetensors metadata"),
    ],
)
def test_selected_reader_rejects_missing_or_corrupt_artifact(
    case: str,
    match: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    path = tmp_path / f"selected-{case}.safetensors"
    if case == "corrupt":
        path.write_bytes(b"not-safetensors")

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module._validate_selected_file(
            path,
            expected_bytes=None,
            expected_metadata=_selected_metadata(contract),
            contract=contract,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        # Each case pins the SPECIFIC sub-check it targets. A shared
        # "selected prefix metadata" prefix would be satisfied by any of the
        # four sub-checks, so swapping two entries of `expected_values` (a
        # plausible copy-paste bug) left all of these green and was caught only
        # incidentally, elsewhere. A test that cannot localise the defect it
        # exists to find makes the next regression harder to root-cause.
        ("missing_key", r"keys mismatch: missing=\['raw_sha256'\]"),
        ("extra_key", r"keys mismatch: .*extra=\['unexpected'\]"),
        ("schema", "field schema_version mismatch"),
        ("kind", "field kind mismatch"),
        ("fingerprint", "field feature_contract_fingerprint mismatch"),
        ("prefix_hash", "invalid prefix_hash"),
        ("prefix_text_sha256", "invalid prefix_text_sha256"),
        ("extraction_identity_sha256", "invalid extraction_identity_sha256"),
        ("raw_sha256", "invalid raw_sha256"),
        ("generator_state_after_sha256", "invalid generator_state_after_sha256"),
    ],
)
def test_selected_writer_rejects_invalid_metadata(
    case: str,
    match: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    metadata = _selected_metadata(contract)
    if case == "missing_key":
        metadata.pop("raw_sha256")
    elif case == "extra_key":
        # The writer has no "expected metadata" to diff against -- unlike the
        # reader, which compares to a caller-supplied dict -- so this key-set
        # guard is the ONLY thing rejecting a spurious key on the write path.
        metadata["unexpected"] = "value"
    elif case == "schema":
        metadata["schema_version"] = "2"
    elif case == "kind":
        metadata["kind"] = "wrong"
    elif case == "fingerprint":
        metadata["feature_contract_fingerprint"] = _SHA_D
    elif case == "prefix_hash":
        metadata["prefix_hash"] = "not-a-prefix"
    elif case in {
        "prefix_text_sha256",
        "extraction_identity_sha256",
        "raw_sha256",
        "generator_state_after_sha256",
    }:
        metadata[case] = "not-a-digest"
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)

    with pytest.raises(AxBenchBpbAdvError, match=match):
        cache_module._serialize_selected_prefix(
            _valid_materialized_segments(),
            metadata=metadata,
            contract=contract,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "value"])
def test_selected_prefix_metadata_key_set_and_values_are_exact(
    mutation: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    expected = _selected_metadata(contract)
    actual = dict(expected)
    if mutation == "missing":
        actual.pop("raw_sha256")
    elif mutation == "extra":
        actual["extra"] = "unexpected"
    elif mutation == "value":
        actual["raw_sha256"] = _SHA_D
    path = tmp_path / f"selected-{mutation}.safetensors"
    save_file(_valid_selected_tensors(), str(path), metadata=actual)

    with pytest.raises(AxBenchBpbAdvError, match="metadata mismatch"):
        cache_module._validate_selected_file(
            path,
            expected_bytes=None,
            expected_metadata=expected,
            contract=contract,
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("bounds", "char bounds"),
        ("misaligned", "misaligned"),
        ("non_rep", "consolidated-away"),
        ("duplicate", "duplicate"),
        ("bucket_range", "out-of-range bucket"),
        ("n_positive", "positive count"),
        ("n_negative", "negative count"),
        ("pct_dtype", "positive percentiles"),
        ("pct_shape", "positive percentiles"),
        ("positive_bucket", "positive bucket"),
        ("negative_bucket", "negative bucket"),
    ],
)
def test_selected_writer_rejects_every_reader_invalid_materialization(
    case: str,
    match: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    first, second = _valid_materialized_segments()
    context = first.context
    if case == "bounds":
        second = replace(
            second,
            context=replace(second.context, char_start=3),
        )
    elif case == "misaligned":
        context = replace(context, buckets=(0,))
    elif case == "non_rep":
        context = replace(context, feature_ids=(2, 4))
    elif case == "duplicate":
        context = replace(context, feature_ids=(1, 1))
    elif case == "bucket_range":
        context = replace(context, buckets=(0, 2))
    elif case == "n_positive":
        first = replace(first, n_positive=3)
    elif case == "n_negative":
        first = replace(first, n_negative=0)
    elif case == "pct_dtype":
        first = replace(
            first,
            positive_percentiles_uint8=t.tensor([204], dtype=t.int64),
        )
    elif case == "pct_shape":
        first = replace(
            first,
            positive_percentiles_uint8=t.tensor([204, 153], dtype=t.uint8),
        )
    elif case == "positive_bucket":
        context = replace(context, buckets=(1, 1))
    elif case == "negative_bucket":
        context = replace(context, buckets=(0, 0))
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)
    first = replace(first, context=context)

    with pytest.raises(Exception, match=match):
        cache_module._serialize_selected_prefix(
            (first, second),
            metadata=_selected_metadata(contract),
            contract=contract,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_ids", (1.5, 4)),
        ("buckets", (False, 1)),
    ],
)
def test_selected_context_rejects_non_integer_elements_at_construction(
    field: str,
    value: tuple[object, ...],
) -> None:
    first, _second = _valid_materialized_segments()

    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        replace(first.context, **{field: value})


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("n_segments_shape", "malformed n_segments"),
        ("n_segments_dtype", "malformed n_segments"),
        ("negative_n_segments", "negative n_segments"),
        ("missing_tensor", "missing tensor"),
        ("extra_tensor", "tensor keys mismatch"),
        ("ids_shape", "malformed IDs"),
        ("ids_dtype", "malformed IDs"),
        ("buckets_shape", "malformed buckets"),
        ("buckets_dtype", "malformed buckets"),
        ("positive_count_shape", "malformed positive count"),
        ("positive_count_dtype", "malformed positive count"),
        ("positive_count_value", "invalid positive count"),
        ("percentiles_shape", "malformed percentiles"),
        ("percentiles_dtype", "malformed percentiles"),
        ("bounds_shape", "malformed char bounds"),
        ("bounds_dtype", "malformed char bounds"),
        ("bounds_order", "invalid char bounds"),
        ("non_rep", "consolidated-away"),
        ("duplicate", "duplicate IDs"),
        ("bucket_range", "invalid buckets"),
        ("positive_bucket", "positive bucket is invalid"),
        ("negative_bucket", "negative bucket is invalid"),
    ],
)
def test_selected_reader_rejects_malformed_and_semantically_invalid_tensors(
    case: str,
    match: str,
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    tensors = _valid_selected_tensors()
    if case == "n_segments_shape":
        tensors["n_segments"] = t.tensor([[2]], dtype=t.int64)
    elif case == "n_segments_dtype":
        tensors["n_segments"] = t.tensor([2], dtype=t.int32)
    elif case == "negative_n_segments":
        tensors = {"n_segments": t.tensor([-1], dtype=t.int64)}
    elif case == "missing_tensor":
        tensors.pop("seg_0_feature_ids")
    elif case == "extra_tensor":
        tensors["extra"] = t.tensor([1])
    elif case == "ids_shape":
        tensors["seg_0_feature_ids"] = t.tensor([[1, 4]], dtype=t.int64)
    elif case == "ids_dtype":
        tensors["seg_0_feature_ids"] = t.tensor([1, 4], dtype=t.int32)
    elif case == "buckets_shape":
        tensors["seg_0_buckets"] = t.tensor([0], dtype=t.int64)
    elif case == "buckets_dtype":
        tensors["seg_0_buckets"] = t.tensor([0, 1], dtype=t.int32)
    elif case == "positive_count_shape":
        tensors["seg_0_n_positive"] = t.tensor([[1]], dtype=t.int64)
    elif case == "positive_count_dtype":
        tensors["seg_0_n_positive"] = t.tensor([1], dtype=t.int32)
    elif case == "positive_count_value":
        tensors["seg_0_n_positive"] = t.tensor([3], dtype=t.int64)
    elif case == "percentiles_shape":
        tensors["seg_0_positive_percentiles_uint8"] = t.tensor(
            [204, 153],
            dtype=t.uint8,
        )
    elif case == "percentiles_dtype":
        tensors["seg_0_positive_percentiles_uint8"] = t.tensor(
            [204],
            dtype=t.int64,
        )
    elif case == "bounds_shape":
        tensors["seg_0_char_bounds"] = t.tensor([0], dtype=t.int64)
    elif case == "bounds_dtype":
        tensors["seg_0_char_bounds"] = t.tensor([0, 4], dtype=t.int32)
    elif case == "bounds_order":
        tensors["seg_1_char_bounds"] = t.tensor([3, 4], dtype=t.int64)
    elif case == "non_rep":
        tensors["seg_0_feature_ids"] = t.tensor([2, 4], dtype=t.int64)
    elif case == "duplicate":
        tensors["seg_0_feature_ids"] = t.tensor([1, 1], dtype=t.int64)
    elif case == "bucket_range":
        tensors["seg_0_buckets"] = t.tensor([0, 2], dtype=t.int64)
    elif case == "positive_bucket":
        tensors["seg_0_buckets"] = t.tensor([1, 1], dtype=t.int64)
    elif case == "negative_bucket":
        tensors["seg_0_buckets"] = t.tensor([0, 0], dtype=t.int64)
    else:  # pragma: no cover - the parameter table owns this domain
        raise AssertionError(case)
    path = tmp_path / f"malformed-{case}.safetensors"
    save_file(tensors, str(path), metadata=_selected_metadata(contract))

    with pytest.raises(Exception, match=match):
        cache_module._validate_selected_file(
            path,
            expected_bytes=None,
            expected_metadata=_selected_metadata(contract),
            contract=contract,
        )


def test_g3b_d3_selected_file_digest_and_counts_come_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A->B->A: the returned digest and counts must describe the SAME bytes.

    `_validate_selected_file` used to read the pathname three times -- bytes for
    the digest, `safe_open` for metadata, `load_file` for tensors -- with no
    invariant binding them. A replacement after the first read made it return
    the FIRST read's sha256 alongside the THIRD read's structural counts, and
    G3c records both in the immutable manifest, so the mismatch surfaces only
    after the run tag is committed.

    Held at the exact observable schedule of that swap: B sits at the pathname
    while the single read returns A. The digest and every count must be A's.
    """
    contract = _contract(tmp_path)
    metadata = _selected_metadata(contract)

    a_bytes = cache_module._serialize_selected_prefix(
        _valid_materialized_segments(),
        metadata=metadata,
        contract=contract,
    )
    single_segment = (_valid_materialized_segments()[0],)
    b_bytes = cache_module._serialize_selected_prefix(
        single_segment,
        metadata=metadata,
        contract=contract,
    )
    assert a_bytes != b_bytes

    path = tmp_path / "selected.safetensors"
    path.write_bytes(b_bytes)

    real_read_bytes = Path.read_bytes

    def _swapped(self: Path) -> bytes:
        if self == path:
            return a_bytes
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _swapped)

    result = cache_module._validate_selected_file(
        path,
        expected_bytes=a_bytes,
        expected_metadata=metadata,
        contract=contract,
    )

    # Every field must describe A. Reopening the pathname would pair A's digest
    # with B's single-segment counts.
    assert result == {
        "selected_sha256": hashlib.sha256(a_bytes).hexdigest(),
        "n_segments": 2,
        "n_features_total": 2,
        "n_positive_total": 1,
        "n_negative_total": 1,
    }


def _with_sampling_positive_selection(contract):
    """Swap the contract's positive selection from threshold- to sampling-shaped.

    ``parse_feature_selection`` dispatches on the key: ``distribution`` yields a
    ``SamplingSelection``, ``threshold_type`` a ``ThresholdSelection``. Every
    other reconstruction test in this file (and in the frozen oracle's own test
    file) uses the threshold shape, so this is what reaches the sampling half of
    ``build_feature_pipeline_from_cfg_block``.

    ``delta_weights`` stays unset: ``reconstruct_clustered_replay_state`` always
    passes ``feature_delta=None``, and the builder raises if exactly one of the
    two is set.
    """
    return replace(
        contract,
        selection=replace(
            contract.selection,
            positive_selection_json=json.dumps(
                {
                    "distribution": "geometric",
                    "mean": 5.0,
                    "std": 5.0,
                    "min_k": 1,
                    "max_k": 2,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def _with_delta_weighted_sampling_positive_selection(contract):
    sampling_contract = _with_sampling_positive_selection(contract)
    positive_selection = json.loads(
        sampling_contract.selection.positive_selection_json
    )
    positive_selection["delta_weights"] = "percentile_delta"
    return replace(
        sampling_contract,
        selection=replace(
            sampling_contract.selection,
            positive_selection_json=json.dumps(
                positive_selection,
                sort_keys=True,
                separators=(",", ":"),
            ),
            feature_delta_json=json.dumps(
                {
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": 1.0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
    )


def test_reconstruct_builds_the_sampling_branch_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the SamplingSelection reconstruction path end to end.

    The oracle's own test file never imports SamplingSelection and pins a
    threshold-shaped selection everywhere, so this branch -- and the second
    ``build_percentile_table`` it needs, since ``ThresholdSelectionConfig``
    carries no ``percentile_table`` -- was reconstructed by no test on either
    side. Instrumenting the whole gate showed 501 ``build_percentile_table``
    calls and zero reached from this function's frame.
    """
    payload = b"one-verified-statistics-snapshot"
    contract = _with_sampling_positive_selection(
        _with_statistics_path(
            _contract(tmp_path),
            "virtual-statistics.safetensors",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )
    fake_stats = SimpleNamespace(
        metadata=SimpleNamespace(n_sae_features=8),
        per_feature_stats=_per_feature_stats(),
    )
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: payload)
    monkeypatch.setattr(
        cache_module,
        "load_statistics_and_checksum_from_bytes",
        lambda snapshot: (fake_stats, _SHA_B, None),
    )
    monkeypatch.setattr(
        cache_module,
        "compute_percentile_checksum",
        lambda _table: _SHA_B,
    )

    state = cache_module.reconstruct_clustered_replay_state(contract)

    # The sampling branch must yield a SamplingBundle carrying its own
    # percentile table -- the capability that makes the second
    # build_percentile_table load-bearing rather than dead.
    selection_config = state.pipeline.selection_config
    assert isinstance(selection_config, SamplingBundle)
    assert selection_config.percentile_table is not None
    assert selection_config.percentile_table.shape[0] == 8
    # The shared-RNG invariant is what distinguishes this branch from the
    # threshold one, where the negative bundle gets its own generator.
    assert state.pipeline.negative_bundle is not None
    assert state.pipeline.negative_bundle.rng is selection_config.rng


def test_reconstruct_threads_checkpoint_delta_into_delta_weighted_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint contract, not prompt-cache config, owns replay delta."""

    payload = b"one-verified-statistics-snapshot"
    contract = _with_delta_weighted_sampling_positive_selection(
        _with_statistics_path(
            _contract(tmp_path),
            "virtual-statistics.safetensors",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    )
    fake_stats = SimpleNamespace(
        metadata=SimpleNamespace(n_sae_features=8),
        per_feature_stats=_per_feature_stats(),
    )
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: payload)
    monkeypatch.setattr(
        cache_module,
        "load_statistics_and_checksum_from_bytes",
        lambda snapshot: (fake_stats, _SHA_B, None),
    )
    monkeypatch.setattr(
        cache_module,
        "compute_percentile_checksum",
        lambda _table: _SHA_B,
    )

    state = cache_module.reconstruct_clustered_replay_state(contract)

    selection_config = state.pipeline.selection_config
    assert isinstance(selection_config, SamplingBundle)
    assert (
        selection_config.selection.delta_weights
        is DeltaSamplingWeights.percentile_delta
    )
    assert state.pipeline.delta_bundle is not None
    assert state.pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual
    assert state.pipeline.delta_bundle.config.neg_delta_ratio == 1.0
