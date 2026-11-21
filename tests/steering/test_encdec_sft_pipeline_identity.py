"""Golden identity surface for the conditioned-SFT data generator."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t

from errors import DataLoadError
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder import sft_selection
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.sft_features import (
    SFTFeatureStore,
    load_sft_features,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import (
    LOWEST_MAGNITUDE_BUCKET,
    MissingPercentilePolicy,
    MissingPercentileResolver,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.feature_io import save_features
from steering.refined_types import PositiveInt
from steering.sft.loss_mask import LossMaskMode
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _full_sequence_char_count,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_sft_features import _write_parquet

N_SAE_FEATURES = 64
N_SHARDS = 3
DOCS_PER_SHARD = 4
N_SEGMENTS = 3
FEATURES_PER_SEGMENT = 6
N_BATCHES = 6
GOLDEN_PATH = Path(__file__).parent / "goldens" / "encdec_sft_pipeline_identity_golden.json"


@dataclass(frozen=True)
class ScenarioSpec:
    capture: Callable[[Path], list[str]]


@dataclass(frozen=True)
class _Family:
    task: Task
    tokenizer: SFTMockTokenizer
    feature_dir: Path
    parquet_dir: Path


@dataclass(frozen=True)
class _Pipeline:
    kwargs: dict[str, Any]

    @property
    def generators(self) -> dict[str, t.Generator]:
        """Discover every digested RNG from the live scenario bundles."""
        output: dict[str, t.Generator] = {}
        sampling = self.kwargs.get("sampling_bundle")
        if isinstance(sampling, SamplingBundle):
            output["sampling"] = sampling.rng
        negative = self.kwargs.get("negative_bundle")
        if isinstance(negative, NegativeBundle):
            output["negative"] = negative.rng
        return output


class _VariableLengthTask(Task):
    """Deterministic mix of skipped, truncated, and ordinary conversations."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._n = n
        self.served: list[int] = []

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        self.served.append(index)
        if index % 6 == 0:
            user, assistant = "u", "a"
        elif index % 6 == 1:
            user, assistant = "long-user", "L" * 90
        else:
            user = f"question-{index:02d}"
            assistant = f"answer-{index:02d}-" + "x" * (8 + index % 5)
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }


def _hash_field(hasher: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    hasher.update(len(label_bytes).to_bytes(8, "little"))
    hasher.update(label_bytes)
    hasher.update(len(payload).to_bytes(8, "little"))
    hasher.update(payload)


def _tensor_payload(value: t.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("conditioned-SFT golden capture requires little-endian tensor bytes")
    is_contiguous = value.is_contiguous()
    cpu = value.detach().cpu().contiguous()
    header = json.dumps(
        {
            "dtype": str(cpu.dtype),
            "is_contiguous": is_contiguous,
            "shape": list(cpu.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = cpu.view(t.uint8).numpy().tobytes()
    return len(header).to_bytes(8, "little") + header + raw


def digest_batch(
    batch: tuple,
    state: sft.SFTState,
    generators: dict[str, t.Generator],
) -> str:
    """Digest every yielded value, cursor field, and distinct RNG state."""
    hasher = hashlib.sha256()
    for index, value in enumerate(batch):
        if value is None:
            _hash_field(hasher, f"batch[{index}]", b"NONE")
        else:
            assert isinstance(value, t.Tensor), type(value)
            _hash_field(hasher, f"batch[{index}]", _tensor_payload(value))
    _hash_field(hasher, "consumed", str(state.consumed).encode("ascii"))
    _hash_field(
        hasher,
        "dataloader_state",
        json.dumps(
            state.dataloader_state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    _hash_field(hasher, "exhausted", b"true" if state.exhausted else b"false")
    _hash_field(hasher, "dataset_size", str(state.dataset_size).encode("ascii"))
    _hash_field(
        hasher,
        "artifact_coverage",
        json.dumps(
            state.artifact_coverage,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    _hash_field(hasher, "generator:torch-default", _tensor_payload(t.random.get_rng_state()))

    roles_by_identity: dict[int, list[str]] = {}
    objects: dict[int, t.Generator] = {}
    for role, generator in sorted(generators.items()):
        identity = id(generator)
        roles_by_identity.setdefault(identity, []).append(role)
        objects[identity] = generator
    distinct = sorted(
        (("+".join(roles), objects[identity]) for identity, roles in roles_by_identity.items()),
        key=lambda item: item[0],
    )
    for roles, generator in distinct:
        _hash_field(hasher, f"generator:{roles}", _tensor_payload(generator.get_state()))
    return hasher.hexdigest()


def _digest_order_contract(batch: tuple, state: sft.SFTState) -> str:
    """Scenario 13's documented resume surface: text tensors plus cursor fields."""
    assert state.dataloader_state is not None
    hasher = hashlib.sha256()
    _hash_field(hasher, "inputs", _tensor_payload(batch[0]))
    _hash_field(hasher, "targets", _tensor_payload(batch[1]))
    cursor = {
        key: state.dataloader_state[key]
        for key in ("consumed", "next_cursor", "buffered_cursor_positions")
    }
    _hash_field(
        hasher,
        "cursor",
        json.dumps(cursor, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return hasher.hexdigest()


def _write_feature_shard(
    family: _Family,
    stem: str,
    original_indices: list[int],
    *,
    singleton_segment_feature_id: int | None = None,
) -> None:
    offsets = [0]
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    doc_ids: list[int] = []
    segment_indices: list[int] = []
    segment_positions: list[int] = []
    char_starts: list[int] = []
    char_ends: list[int] = []
    for local_doc_id, original_index in enumerate(original_indices):
        total_chars = _full_sequence_char_count(
            family.tokenizer,
            family.task[original_index],
        )
        segment_width = max(1, total_chars // N_SEGMENTS)
        for segment_idx in range(N_SEGMENTS):
            if (
                singleton_segment_feature_id is not None
                and original_index == 2
                and segment_idx == 0
            ):
                ids = [singleton_segment_feature_id]
                mags = [12.0]
            else:
                start = (original_index * 11 + segment_idx * 7) % N_SAE_FEATURES
                ids = sorted(
                    (start + offset * 5) % N_SAE_FEATURES for offset in range(FEATURES_PER_SEGMENT)
                )
                mags = [
                    1.0 + 1.5 * offset + 0.25 * (original_index % 3) for offset in range(len(ids))
                ]
            feature_ids.extend(ids)
            feature_mags.extend(mags)
            offsets.append(len(feature_ids))
            doc_ids.append(local_doc_id)
            segment_indices.append(segment_idx)
            segment_positions.append(segment_idx * 128)
            char_starts.append(segment_idx * segment_width)
            char_ends.append(
                total_chars if segment_idx == N_SEGMENTS - 1 else (segment_idx + 1) * segment_width
            )
    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(segment_indices, dtype=t.int32),
        "segment_positions": t.tensor(segment_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * len(original_indices), dtype=t.int32),
        "n_docs": len(original_indices),
        "n_segments": len(doc_ids),
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    save_features(shard, family.feature_dir / f"{stem}_features")
    _write_parquet(family.parquet_dir, stem, original_indices)


def _build_family(
    root: Path,
    *,
    split: SFTSplit = SFTSplit.TRAIN,
    singleton_segment_feature_id: int | None = None,
) -> _Family:
    tokenizer = SFTMockTokenizer()
    task = _VariableLengthTask(N_SHARDS * DOCS_PER_SHARD)
    feature_dir = root / "features"
    parquet_dir = root / "parquet"
    feature_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)
    family = _Family(task, tokenizer, feature_dir, parquet_dir)
    for shard_idx in range(N_SHARDS):
        start = shard_idx * DOCS_PER_SHARD
        _write_feature_shard(
            family,
            f"shard_{shard_idx:05d}",
            list(range(start, start + DOCS_PER_SHARD)),
            singleton_segment_feature_id=singleton_segment_feature_id,
        )
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(
            {
                "mixture": "capabilities",
                "split": split.value,
                "format": "full_nanochat_text",
                "completed_shards": N_SHARDS,
                "total_shards": N_SHARDS,
                "total_conversations": len(task),
            }
        ),
        encoding="utf-8",
    )
    task.served.clear()
    return family


def _table(*, uncovered: tuple[int, ...] = (), offset: float = 0.0) -> t.Tensor:
    table = (
        t.linspace(0.5 + offset, 12.5 + offset, 99, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .clone()
    )
    if uncovered:
        table[list(uncovered)] = 0.0
    return table


def _lowest_resolver(uncovered: tuple[int, ...]) -> MissingPercentileResolver:
    covered = t.ones(N_SAE_FEATURES, dtype=t.bool)
    covered[list(uncovered)] = False
    return MissingPercentileResolver(MissingPercentilePolicy.LOWEST, covered)


def _boundaries(*, binary: bool = False, width: int = N_SAE_FEATURES) -> t.Tensor:
    if binary:
        return t.zeros(width, 0, dtype=t.float64)
    return t.tensor([2.5, 5.0, 7.5], dtype=t.float64).expand(width, -1).contiguous()


def _sampling_bundle(
    *,
    seed: int,
    mode: SamplingMode = SamplingMode.weighted,
    table: t.Tensor | None = None,
) -> SamplingBundle:
    return SamplingBundle(
        selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=5,
            mode=mode,
        ),
        percentile_table=_table() if table is None else table,
        rng=t.Generator().manual_seed(seed),
    )


def _negative_bundle(
    *,
    seed: int,
    rng: t.Generator | None = None,
    feature_mask: FeatureMask | None = None,
) -> NegativeBundle:
    return NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=4,
        ),
        n_sae_features=N_SAE_FEATURES,
        rng=t.Generator().manual_seed(seed) if rng is None else rng,
        feature_mask=feature_mask,
    )


def _consolidation(root: Path) -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((1, 6, 11), (16, 21), (26, 31, 36), (41, 46)))
    resolved = ResolvedFeatureClusters(
        source_path=root / "clusters.yaml",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=N_SAE_FEATURES,
        ),
    )
    return FeatureClusterConsolidation.build(resolved, None)


def _threshold_pipeline(
    threshold_type: ThresholdType = ThresholdType.segment_topk,
    threshold: float = 4,
) -> _Pipeline:
    return _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(threshold_type, threshold),
            "boundaries": _boundaries(),
        },
    )


def _make_generator(
    family: _Family,
    pipeline: _Pipeline,
    state: sft.SFTState,
    **overrides: Any,
) -> Iterator[tuple]:
    kwargs: dict[str, Any] = {
        "task": family.task,
        "tokenizer": family.tokenizer,
        "B": 2,
        "T": 96,
        "device": "cpu",
        "sft_state": state,
        "buffer_size": 4,
        "grad_accum_steps": 1,
        "feature_dir": str(family.feature_dir),
        "parquet_dir": str(family.parquet_dir),
        "min_loss_tokens": 8,
        "alignment_probe_samples": 0,
        "expected_split": SFTSplit.TRAIN,
        **pipeline.kwargs,
        **overrides,
    }
    return sft.make_sft_data_generator(**kwargs)


@contextmanager
def _open_store(family: _Family) -> Iterator[SFTFeatureStore]:
    store = load_sft_features(str(family.feature_dir), str(family.parquet_dir))
    with (
        patch.object(sft, "load_sft_features", return_value=store),
        patch.object(sft, "_assert_sft_mixture_fingerprint_matches_task", return_value=None),
    ):
        yield store
    store._feature_index.close()


def _capture_standard(
    root: Path,
    pipeline: _Pipeline,
    *,
    batches: int = N_BATCHES,
    singleton_segment_feature_id: int | None = None,
    inspect_batch: Callable[[tuple], None] | None = None,
    **overrides: Any,
) -> list[str]:
    family = _build_family(
        root,
        singleton_segment_feature_id=singleton_segment_feature_id,
    )
    assert isinstance(family.task, _VariableLengthTask)
    state = sft.SFTState()
    packed_multiple_conversations = False
    fp_cross_document = False
    digests: list[str] = []
    with _open_store(family) as store:
        generator = _make_generator(
            family,
            pipeline,
            state,
            **{"return_seg_doc_id": True, **overrides},
        )
        for _ in range(batches):
            batch = next(generator)
            assert len(batch) == 8
            packed_multiple_conversations |= int(batch[7].max().item()) >= 1
            if pipeline.kwargs.get("fp_percentile_table") is not None:
                assert batch[6] is not None
                fp_cross_document |= int(batch[7].max().item()) >= 1
            if inspect_batch is not None:
                inspect_batch(batch)
            digests.append(digest_batch(batch, state, pipeline.generators))
        assert store._feature_index.cache_stats.loads >= N_SHARDS
    assert packed_multiple_conversations
    if pipeline.kwargs.get("fp_percentile_table") is not None:
        assert fp_cross_document
    assert 0 in family.task.served and 1 in family.task.served
    return digests


def _scenario_threshold_topk_negatives_fp_lowest(root: Path) -> list[str]:
    uncovered = tuple(range(0, N_SAE_FEATURES, 11))
    negative = _negative_bundle(seed=1504)
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(
                ThresholdType.feature_percentile_segment_topk,
                3.50,
            ),
            "precomputed_thresholds": t.full((N_SAE_FEATURES,), 3.0, dtype=t.float64),
            "boundaries": _boundaries(binary=True),
            "negative_bundle": negative,
            "fp_percentile_table": _table(uncovered=uncovered),
            "n_sae_features": N_SAE_FEATURES,
            "missing_percentile": _lowest_resolver(uncovered),
        },
    )
    return _capture_standard(
        root,
        pipeline,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
        return_seg_doc_id=True,
    )


def _scenario_threshold_precomputed_masked_negatives(root: Path) -> list[str]:
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE_FEATURES, 8)))
    allowed = set(mask.ids)
    assert any(
        not allowed.intersection(
            (original_index * 11 + segment_idx * 7 + offset * 5) % N_SAE_FEATURES
            for offset in range(FEATURES_PER_SEGMENT)
        )
        for original_index in range(N_SHARDS * DOCS_PER_SHARD)
        for segment_idx in range(N_SEGMENTS)
    )
    negative = _negative_bundle(seed=1505, feature_mask=mask)
    thresholds = t.full((N_SAE_FEATURES,), 100.0, dtype=t.float64)
    thresholds[list(mask.ids)] = 4.0
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(ThresholdType.feature_percentile, 0.5),
            "precomputed_thresholds": thresholds,
            "boundaries": _boundaries(),
            "feature_mask": mask,
            "negative_bundle": negative,
        },
    )
    return _capture_standard(root, pipeline)


def _scenario_sampling_weighted_fp_shared_table_masked(root: Path) -> list[str]:
    table = _table()
    sampling = _sampling_bundle(seed=1506, table=table)
    mask = FeatureMask(FeatureMaskMode.blacklist, (0, 2, 4, 6, 8))
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "boundaries": _boundaries(),
            "feature_mask": mask,
            "fp_percentile_table": table,
            "n_sae_features": N_SAE_FEATURES,
        },
    )
    assert sampling.percentile_table is pipeline.kwargs["fp_percentile_table"]
    return _capture_standard(root, pipeline, return_seg_doc_id=True)


def _scenario_sampling_topk_percentile(root: Path) -> list[str]:
    sampling = _sampling_bundle(seed=1507, mode=SamplingMode.top_k_percentile)
    pipeline = _Pipeline(
        kwargs={"sampling_bundle": sampling, "boundaries": _boundaries()},
    )
    return _capture_standard(root, pipeline)


def _scenario_sampling_weighted_negatives_shared_rng(root: Path) -> list[str]:
    sampling = _sampling_bundle(seed=1508)
    negative = _negative_bundle(seed=0, rng=sampling.rng)
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "negative_bundle": negative,
            "boundaries": _boundaries(),
        },
    )
    assert sampling.rng is negative.rng
    return _capture_standard(root, pipeline)


def _scenario_sampling_lowest_own_table(root: Path) -> list[str]:
    uncovered = tuple(range(3, N_SAE_FEATURES, 13))
    sampling_table = _table(uncovered=uncovered)
    fp_table = _table(offset=0.25)
    sampling = _sampling_bundle(seed=1509, table=sampling_table)
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "boundaries": _boundaries(),
            "fp_percentile_table": fp_table,
            "n_sae_features": N_SAE_FEATURES,
            "missing_percentile": _lowest_resolver(uncovered),
        },
    )
    assert sampling.percentile_table is not fp_table
    uncovered_id = uncovered[0]
    observed_uncovered = False

    def assert_lowest_bucket(batch: tuple) -> None:
        nonlocal observed_uncovered
        selected = batch[2] == uncovered_id
        if bool(selected.any()):
            observed_uncovered = True
            assert bool((batch[3][selected] == LOWEST_MAGNITUDE_BUCKET).all())

    digests = _capture_standard(
        root,
        pipeline,
        singleton_segment_feature_id=uncovered_id,
        inspect_batch=assert_lowest_bucket,
        return_seg_doc_id=True,
    )
    assert observed_uncovered
    return digests


def _scenario_clustered_threshold_packed_plain(root: Path) -> list[str]:
    consolidation = _consolidation(root)
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(
                ThresholdType.feature_percentile,
                0.50,
            ),
            "boundaries": _boundaries(),
            "consolidation": consolidation,
            "cluster_threshold_percentile_table": _table(),
        },
    )
    return _capture_standard(root, pipeline)


def _scenario_clustered_threshold_packed_topk(root: Path) -> list[str]:
    # The combined cutoff/top-k arm (feature_percentile_segment_topk) of
    # select_from_packed_percentiles is a distinct implementation from the
    # plain-percentile arm above; both need identity coverage.
    consolidation = _consolidation(root)
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(
                ThresholdType.feature_percentile_segment_topk,
                3.50,
            ),
            "boundaries": _boundaries(),
            "consolidation": consolidation,
            "cluster_threshold_percentile_table": _table(),
        },
    )
    return _capture_standard(root, pipeline)


def _scenario_clustered_threshold_plain(root: Path) -> list[str]:
    consolidation = _consolidation(root)
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(ThresholdType.segment_topk, 2),
            "boundaries": _boundaries(),
            "consolidation": consolidation,
            "cluster_threshold_percentile_table": _table(),
        },
    )
    return _capture_standard(root, pipeline)


def _scenario_clustered_sampling_negatives(root: Path) -> list[str]:
    consolidation = _consolidation(root)
    fp_table = _table()
    sampling = _sampling_bundle(seed=1510, table=fp_table)
    negative = _negative_bundle(
        seed=0,
        rng=sampling.rng,
        feature_mask=consolidation.effective_negative_mask(),
    )
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "negative_bundle": negative,
            "boundaries": _boundaries(),
            "consolidation": consolidation,
            "fp_percentile_table": fp_table,
            "n_sae_features": N_SAE_FEATURES,
        },
    )
    assert sampling.percentile_table is pipeline.kwargs["fp_percentile_table"]
    original_sample_features = sft_selection.sample_features
    original_ranks = MissingPercentileResolver.ranks
    override_weights_seen: list[bool] = []
    rank_calls = 0

    def record_sample_features(*args: Any, **kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
        override_weights_seen.append(kwargs.get("override_weights") is not None)
        return original_sample_features(*args, **kwargs)

    def record_ranks(self: MissingPercentileResolver, *args: Any, **kwargs: Any) -> t.Tensor:
        nonlocal rank_calls
        rank_calls += 1
        return original_ranks(self, *args, **kwargs)

    with (
        patch.object(sft_selection, "sample_features", side_effect=record_sample_features),
        patch.object(MissingPercentileResolver, "ranks", record_ranks),
    ):
        digests = _capture_standard(root, pipeline)
    assert override_weights_seen and all(override_weights_seen)
    assert rank_calls * N_SEGMENTS == len(override_weights_seen)
    return digests


def _scenario_compact_fp_layout(root: Path) -> list[str]:
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(0, N_SAE_FEATURES, 2)))
    vocab = effective_feature_vocab(mask, N_SAE_FEATURES)
    assert vocab is not None
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(N_SAE_FEATURES),
        feature_mask=mask,
        feature_clusters=None,
        feature_vocab=vocab,
    )
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(ThresholdType.segment_topk, 4),
            "boundaries": _boundaries(),
            "feature_mask": mask,
            "fp_percentile_table": _table(),
            "fp_target_layout": layout,
        },
    )
    return _capture_standard(root, pipeline, return_seg_doc_id=True)


def _scenario_train_val_interleave(root: Path) -> list[str]:
    train_family = _build_family(root / "train", split=SFTSplit.TRAIN)
    val_family = _build_family(root / "val", split=SFTSplit.VAL)
    sampling = _sampling_bundle(seed=1511)
    negative = _negative_bundle(seed=0, rng=sampling.rng)
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "negative_bundle": negative,
            "boundaries": _boundaries(),
        },
    )
    train_store = load_sft_features(str(train_family.feature_dir), str(train_family.parquet_dir))
    val_store = load_sft_features(str(val_family.feature_dir), str(val_family.parquet_dir))

    def resolve_store(feature_dir: str, _parquet_dir: str) -> SFTFeatureStore:
        return train_store if feature_dir == str(train_family.feature_dir) else val_store

    train_state = sft.SFTState()
    val_state = sft.SFTState()
    with (
        patch.object(sft, "load_sft_features", side_effect=resolve_store),
        patch.object(sft, "_assert_sft_mixture_fingerprint_matches_task", return_value=None),
    ):
        train = _make_generator(train_family, pipeline, train_state)
        val = _make_generator(
            val_family,
            pipeline,
            val_state,
            single_pass=True,
            expected_split=SFTSplit.VAL,
        )
        digests = [digest_batch(next(train), train_state, pipeline.generators) for _ in range(2)]
        while not val_state.exhausted:
            digests.append(digest_batch(next(val), val_state, pipeline.generators))
        digests.append(digest_batch(next(val), val_state, pipeline.generators))
        digests.extend(
            digest_batch(next(train), train_state, pipeline.generators) for _ in range(2)
        )
    train_store._feature_index.close()
    val_store._feature_index.close()
    return digests


def _scenario_resume_roundtrip_deterministic(root: Path) -> list[str]:
    family = _build_family(root)
    pipeline = _threshold_pipeline()
    with _open_store(family):
        state = sft.SFTState()
        generator = _make_generator(family, pipeline, state)
        first_digests: list[str] = []
        for _ in range(2):
            first_digests.append(digest_batch(next(generator), state, {}))
        resume_state = sft.SFTState(
            consumed=state.consumed,
            dataloader_state=copy.deepcopy(state.dataloader_state),
        )
        resumed = _make_generator(family, pipeline, resume_state)

        reference_state = sft.SFTState()
        reference = _make_generator(family, pipeline, reference_state)
        next(reference)
        next(reference)
        resumed_digests: list[str] = []
        reference_digests: list[str] = []
        for _ in range(2):
            actual = next(resumed)
            resumed_digests.append(digest_batch(actual, resume_state, {}))
            expected = next(reference)
            reference_digests.append(digest_batch(expected, reference_state, {}))
            assert len(actual) == len(expected)
            for actual_value, expected_value in zip(actual, expected, strict=True):
                if actual_value is None:
                    assert expected_value is None
                else:
                    assert t.equal(actual_value, expected_value)
        assert resumed_digests == reference_digests
    return first_digests + resumed_digests


def _scenario_resume_roundtrip_stochastic_order(root: Path) -> list[str]:
    family = _build_family(root)

    def make_pipeline() -> _Pipeline:
        negative = _negative_bundle(seed=1512)
        return _Pipeline(
            kwargs={
                "threshold_cfg": ThresholdConfig(ThresholdType.segment_topk, 4),
                "boundaries": _boundaries(),
                "negative_bundle": negative,
            },
        )

    with _open_store(family):
        initial_pipeline = make_pipeline()
        state = sft.SFTState()
        generator = _make_generator(family, initial_pipeline, state)
        first_digests: list[str] = []
        for _ in range(2):
            first_digests.append(digest_batch(next(generator), state, initial_pipeline.generators))
        resume_state = sft.SFTState(
            consumed=state.consumed,
            dataloader_state=copy.deepcopy(state.dataloader_state),
        )
        resumed_pipeline = make_pipeline()
        resumed = _make_generator(family, resumed_pipeline, resume_state)

        reference_pipeline = make_pipeline()
        reference_state = sft.SFTState()
        reference = _make_generator(family, reference_pipeline, reference_state)
        next(reference)
        next(reference)
        resumed_order: list[str] = []
        reference_order: list[str] = []
        resumed_full: list[str] = []
        for _ in range(2):
            resumed_batch = next(resumed)
            resumed_order.append(_digest_order_contract(resumed_batch, resume_state))
            reference_order.append(_digest_order_contract(next(reference), reference_state))
            # Two distinct surfaces on purpose: the order digests are the
            # resume CONTRACT (conversation order only — the reference stream
            # consumed different RNG, so full tensors legitimately differ),
            # while the resumed stream is still deterministic under its fresh
            # bundle seed, so its full batches are golden-pinned separately to
            # keep the replay path oracle-covered.
            resumed_full.append(
                digest_batch(resumed_batch, resume_state, resumed_pipeline.generators)
            )
        assert resumed_order == reference_order
    return first_digests + resumed_order + resumed_full


def _scenario_threshold_segment_percentile(root: Path) -> list[str]:
    return _capture_standard(
        root,
        _threshold_pipeline(ThresholdType.segment_percentile, 0.5),
    )


def _scenario_ddp_two_rank_striping(root: Path) -> list[str]:
    family = _build_family(root)
    pipeline = _threshold_pipeline()
    output: list[str] = []
    with _open_store(family):
        for rank in (0, 1):
            state = sft.SFTState()
            generator = _make_generator(
                family,
                pipeline,
                state,
                ddp_rank=rank,
                ddp_world_size=2,
            )
            output.extend(
                digest_batch(next(generator), state, pipeline.generators) for _ in range(4)
            )
    return output


SCENARIOS: dict[str, ScenarioSpec] = {
    "threshold_topk_negatives_fp_lowest": ScenarioSpec(
        _scenario_threshold_topk_negatives_fp_lowest
    ),
    "threshold_precomputed_masked_negatives": ScenarioSpec(
        _scenario_threshold_precomputed_masked_negatives
    ),
    "sampling_weighted_fp_shared_table_masked": ScenarioSpec(
        _scenario_sampling_weighted_fp_shared_table_masked
    ),
    "sampling_topk_percentile": ScenarioSpec(_scenario_sampling_topk_percentile),
    "sampling_weighted_negatives_shared_rng": ScenarioSpec(
        _scenario_sampling_weighted_negatives_shared_rng
    ),
    "sampling_lowest_own_table": ScenarioSpec(_scenario_sampling_lowest_own_table),
    "clustered_threshold_packed_plain": ScenarioSpec(_scenario_clustered_threshold_packed_plain),
    "clustered_threshold_packed_topk": ScenarioSpec(_scenario_clustered_threshold_packed_topk),
    "clustered_threshold_plain": ScenarioSpec(_scenario_clustered_threshold_plain),
    "clustered_sampling_negatives": ScenarioSpec(_scenario_clustered_sampling_negatives),
    "compact_fp_layout": ScenarioSpec(_scenario_compact_fp_layout),
    "train_val_interleave": ScenarioSpec(_scenario_train_val_interleave),
    "resume_roundtrip_deterministic": ScenarioSpec(_scenario_resume_roundtrip_deterministic),
    "resume_roundtrip_stochastic_order": ScenarioSpec(_scenario_resume_roundtrip_stochastic_order),
    "threshold_segment_percentile": ScenarioSpec(_scenario_threshold_segment_percentile),
    "ddp_two_rank_striping": ScenarioSpec(_scenario_ddp_two_rank_striping),
}


def _golden() -> Mapping[str, list[str]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def capture_scenario(spec: ScenarioSpec, root: Path) -> list[str]:
    """Run one scenario from a fixed, isolated default torch RNG state."""
    with t.random.fork_rng(devices=[]):
        t.manual_seed(1504)
        return spec.capture(root)


def test_golden_registry_key_set() -> None:
    assert set(_golden()) == set(SCENARIOS)


@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_pipeline_identity_golden(scenario_name: str, tmp_path: Path) -> None:
    assert (
        capture_scenario(SCENARIOS[scenario_name], tmp_path / scenario_name)
        == _golden()[scenario_name]
    )


def _simple_generator(
    task: Task,
    tokenizer: SFTMockTokenizer,
    features: dict[int, list[SegmentFeatures]],
    state: sft.SFTState,
    **overrides: Any,
) -> Iterator[tuple]:
    pipeline = _threshold_pipeline()
    kwargs: dict[str, Any] = {
        "task": task,
        "tokenizer": tokenizer,
        "B": 1,
        "T": 32,
        "device": "cpu",
        "sft_state": state,
        "buffer_size": 2,
        "feature_dir": "/dummy/features",
        "parquet_dir": "/dummy/parquet",
        "min_loss_tokens": 5,
        "alignment_probe_samples": 0,
        "expected_split": SFTSplit.TRAIN,
        **pipeline.kwargs,
        **overrides,
    }
    with patch_load_sft_features(features):
        yield from sft.make_sft_data_generator(**kwargs)


def test_replayed_conversation_min_loss_failure() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=2, user_chars=1, asst_chars=1)
    features = build_synthetic_features(task, tokenizer)
    state = sft.SFTState(
        consumed=0,
        dataloader_state={
            "ddp_rank": 0,
            "ddp_world_size": 1,
            "consumed": 0,
            "next_cursor": 1,
            "buffered_cursor_positions": [0],
            "buffer_size": 2,
            "device_batch_size": 1,
            "sequence_length": 32,
            "grad_accum_steps": 1,
            "min_loss_tokens": 5,
            "loss_mask_mode": LossMaskMode.FULL_SEQUENCE.value,
        },
    )
    generator = _simple_generator(task, tokenizer, features, state)
    with pytest.raises(
        DataLoadError,
        match=(
            r"conditioned SFT buffered conversation no longer satisfies "
            r"min_loss_tokens=5 at logical cursor 0; the checkpoint inputs changed"
        ),
    ):
        next(generator)


def test_buffer_starvation_failure() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=2, user_chars=1, asst_chars=1)
    features = build_synthetic_features(task, tokenizer)
    generator = _simple_generator(task, tokenizer, features, sft.SFTState())
    message = (
        "SFT buffer starvation: scanned all 2 conversations without finding one with >= 5 "
        "supervised loss tokens after truncation to 33 tokens. The min_loss_tokens config "
        "knob is set to 5; lower it, increase max_seq_len (null uses the checkpoint's "
        "sequence_len), or check that the dataset has conversations with enough rendered "
        "tokens. (ddp_rank=0)"
    )
    with pytest.raises(
        RuntimeError,
        match=rf"^{re.escape(message)}$",
    ):
        next(generator)


def test_coverage_gap_failure() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=2, user_chars=4, asst_chars=50)
    features = build_synthetic_features(task, tokenizer)
    features[0][0] = SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=features[0][0].feature_ids,
        feature_mags=features[0][0].feature_mags,
        char_start=0,
        char_end=1,
    )
    generator = _simple_generator(
        task,
        tokenizer,
        features,
        sft.SFTState(),
        min_loss_tokens=1,
        T=96,
    )
    gap = 55
    message = (
        f"conditioned SFT example 0: feature coverage ends {gap} chars before the rendered "
        "conversation. Re-extract full-sequence SFT features with the same rendered task "
        "mixture and tokenization settings before training."
    )
    with pytest.raises(
        ValueError,
        match=rf"^{re.escape(message)}$",
    ):
        next(generator)


def test_truncation_removes_every_segment_failure() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=2, user_chars=4, asst_chars=90)
    features = build_synthetic_features(task, tokenizer)
    for index in range(len(task)):
        total_chars = _full_sequence_char_count(tokenizer, task[index])
        original = features[index][0]
        features[index][0] = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=original.feature_ids,
            feature_mags=original.feature_mags,
            char_start=total_chars - 10,
            char_end=total_chars,
        )
    generator = _simple_generator(
        task,
        tokenizer,
        features,
        sft.SFTState(),
        min_loss_tokens=1,
        T=16,
    )
    message = (
        "conditioned SFT example 0: truncation removed every feature segment. Re-extract "
        "full-sequence SFT features with matching rendering and tokenization settings before "
        "training."
    )
    with pytest.raises(
        ValueError,
        match=rf"^{re.escape(message)}$",
    ):
        next(generator)


def test_compact_fp_source_oob_failure() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=2, user_chars=4, asst_chars=12)
    features = build_synthetic_features(task, tokenizer)
    for index in range(len(task)):
        original = features[index][0]
        features[index][0] = SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=t.tensor([N_SAE_FEATURES], dtype=t.int32),
            feature_mags=t.tensor([4.0], dtype=t.float32),
            char_start=original.char_start,
            char_end=original.char_end,
        )
    compact_vocab = FeatureVocab(
        n_sae_features=N_SAE_FEATURES,
        original_ids=tuple(range(0, N_SAE_FEATURES, 2)),
    )
    original_to_output = t.full((N_SAE_FEATURES,), -1, dtype=t.int64)
    original_to_output[compact_vocab.new_to_old_tensor()] = t.arange(
        compact_vocab.size, dtype=t.int64
    )
    layout = FeaturePredictionTargetLayout(
        n_sae_features=PositiveInt.parse(N_SAE_FEATURES),
        output_width=compact_vocab.size,
        feature_mask=None,
        feature_clusters=None,
        feature_vocab=compact_vocab,
        original_to_output_row=original_to_output,
    )
    pipeline = _Pipeline(
        kwargs={
            "threshold_cfg": ThresholdConfig(ThresholdType.segment_topk, 1),
            "boundaries": _boundaries(width=N_SAE_FEATURES + 1),
            "fp_percentile_table": _table().new_ones((N_SAE_FEATURES + 1, 99)),
            "fp_target_layout": layout,
        },
    )
    with patch_load_sft_features(features):
        generator = _make_generator(
            _Family(task, tokenizer, Path("/dummy/features"), Path("/dummy/parquet")),
            pipeline,
            sft.SFTState(),
            min_loss_tokens=1,
        )
        with pytest.raises(
            ValueError,
            match=rf"SFT FP source feature ID {N_SAE_FEATURES} is outside \[0, {N_SAE_FEATURES}\)",
        ):
            next(generator)


def test_post_exhaustion_repeats_rng_neutral_padding() -> None:
    tokenizer = SFTMockTokenizer()
    task = SingleTurnTask(n=1, user_chars=4, asst_chars=12)
    features = build_synthetic_features(task, tokenizer, n_segments_per_conv=2)
    sampling = _sampling_bundle(seed=1513)
    negative = _negative_bundle(seed=0, rng=sampling.rng)
    pipeline = _Pipeline(
        kwargs={
            "sampling_bundle": sampling,
            "negative_bundle": negative,
            "boundaries": _boundaries(),
        },
    )
    state = sft.SFTState()
    family = _Family(task, tokenizer, Path("/dummy/features"), Path("/dummy/parquet"))
    with patch_load_sft_features(features, split=SFTSplit.VAL):
        generator = _make_generator(
            family,
            pipeline,
            state,
            B=2,
            single_pass=True,
            expected_split=SFTSplit.VAL,
            min_loss_tokens=1,
        )
        next(generator)
        assert state.exhausted
        consumed = state.consumed
        assert state.dataloader_state is not None
        next_cursor = state.dataloader_state["next_cursor"]
        rng_state = sampling.rng.get_state().clone()
        first = next(generator)
        first_digest = digest_batch(first, state, pipeline.generators)
        second = next(generator)
        second_digest = digest_batch(second, state, pipeline.generators)

    assert bool((first[1] == -1).all())
    assert bool((second[1] == -1).all())
    assert first_digest == second_digest
    assert state.consumed == consumed
    assert state.dataloader_state is not None
    assert state.dataloader_state["next_cursor"] == next_cursor
    assert state.exhausted
    assert t.equal(sampling.rng.get_state(), rng_state)
    assert negative.rng is sampling.rng
