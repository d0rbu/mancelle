from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from safetensors.torch import save_file

import scripts.eval.build_eval_context_cache as builder
from errors import FeatureClustersError, RemovedFeatureError
from steering._common.strict_json import canonical_json
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingSelection,
)
from steering.eval.axbench_base.seed_contract import (
    sha256_bytes,
    sha256_file,
)
from steering.eval.axbench_base.stage4_contexts import (
    STAGE4_CONTEXTS_SCHEMA_VERSION,
)
from steering.eval.cluster_feature_contract import ClusterFeatureContractError
from steering.eval.context_cache_io import (
    CLUSTER_SELECTION_SEMANTICS,
    CLUSTER_SELECTION_SEMANTICS_KEY,
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    context_length_cache_dir,
    save_context_cache,
)
from steering.eval.context_enumeration import EvalContextRecord
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.feature_io import save_features
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import compute_tokenizer_checksum
from steering.stats_io import content_sha256, load_statistics, save_percentile_checksum
from steering.stats_types import PerFeatureStat
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.conftest import (
    TEST_CACHE_IDENTITY,
    TEST_CACHE_N_SAE_FEATURES,
    TEST_CACHE_NUM_BUCKETS,
    MockTokenizer,
)
from tests.statistics_doubles import fixed_statistics_double


class VisibleBosTokenizer(MockTokenizer):
    def decode(self, ids: list[int]) -> str:
        parts = []
        for tok in ids:
            if tok == self._bos_id:
                parts.append("<bos>")
            elif tok == self._eos_id:
                parts.append("<eos>")
            else:
                parts.append(chr(tok - 2))
        return "".join(parts)


def _cfg(tmp_path, **overrides):
    data = {
        "stage": "convert",
        "contexts_file": str(tmp_path / "contexts.jsonl"),
        "context_parquet_dir": str(tmp_path / "context_parquet"),
        "feature_dir": str(tmp_path / "features"),
        "output_dir": str(tmp_path / "cache"),
        "n_sae_features": 8,
        "num_buckets": 4,
        "source_bos_char_len": 0,
        "stats_path": str(tmp_path / "stats.safetensors"),
        "feature_selection": {"threshold_type": "segment_topk", "threshold": 1},
        "negative_selection": None,
        "identity": {
            "model_id": "test-model",
            "sae_id": "test-sae",
            "layer": 0,
            "pooling": "mean",
            "feature_selection_hash": "0" * 64,
        },
        "metadata": {"suite": "unit"},
        "shard_size": 1,
        "overwrite": False,
        "zstd_level": None,
        "evals": "core",
        "model_max_seq_len": 1024,
        "num_seeds": 1,
        "base_seed": 3,
        "max_per_task": 1,
        "chat_tasks": None,
        "chat_batch_size": 2,
        "chat_max_problems": 1,
        "harm_num_benign": 1,
        "harm_num_harmful": 1,
    }
    data.update(overrides)
    return OmegaConf.create(data)


def _write_contexts(path):
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "suite": "core",
                        "task": "toy",
                        "seed": 3,
                        "example_index": 4,
                        "row_index": 5,
                        "token_ids": [0, 2, 3],
                        "context_text": "ab",
                        "metadata": {"kind": "mc"},
                    }
                ),
                "",
            ]
        )
    )


def _write_required_stage3e_contract(tmp_path):
    """Write one self-consistent Stage 3e JSONL/parquet/manifest set."""

    committed_dir = tmp_path / "committed_stage3e"
    committed_dir.mkdir()
    token_ids = [0, 99, 100]
    record = EvalContextRecord(
        suite="axbench_base_stage4",
        task=None,
        seed=None,
        example_index=0,
        row_index=None,
        token_ids=tuple(token_ids),
        context_text="ab",
        metadata={},
    )
    jsonl_path = committed_dir / "stage4_contexts.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "suite": record.suite,
                "task": record.task,
                "seed": record.seed,
                "example_index": record.example_index,
                "row_index": record.row_index,
                "token_ids": token_ids,
                "context_text": record.context_text,
                "metadata": {"context_key": record.key, "uses": []},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    parquet_path = committed_dir / "stage4_contexts.parquet"
    pq.write_table(
        pa.table(
            {
                "candidate_index": pa.array([0], type=pa.int64()),
                "text": pa.array([record.context_text], type=pa.string()),
                "context_key": pa.array([record.key], type=pa.string()),
                "token_count": pa.array([len(token_ids)], type=pa.int32()),
            }
        ),
        parquet_path,
    )
    semantics = {
        "schema_version": STAGE4_CONTEXTS_SCHEMA_VERSION,
        "seed_pool_fingerprint": "a" * 64,
        "prompting_description_hashes": {},
        "method_config_fingerprint": "b" * 64,
        "method_config": {},
        "tokenizer_checksum": compute_tokenizer_checksum(MockTokenizer()),
        "encdec_sequence_len": 1024,
        "context_count": 1,
        "context_keys": [record.key],
        "context_token_counts": [len(token_ids)],
    }
    manifest = {
        **semantics,
        "contexts_fingerprint": sha256_bytes(
            canonical_json(semantics).encode("utf-8")
        ),
        "files": {
            "jsonl": {
                "relative_path": jsonl_path.name,
                "sha256": sha256_file(jsonl_path),
            },
            "parquet": {
                "relative_path": parquet_path.name,
                "sha256": sha256_file(parquet_path),
            },
        },
    }
    manifest_path = committed_dir / "stage4_contexts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path, jsonl_path, parquet_path, manifest


def test_load_context_records_validates_jsonl_contract(tmp_path):
    path = tmp_path / "contexts.jsonl"
    _write_contexts(path)

    records = builder.load_context_records(path)

    assert len(records) == 1
    assert records[0].suite == "core"
    assert records[0].task == "toy"
    assert records[0].seed == 3
    assert records[0].example_index == 4
    assert records[0].row_index == 5
    assert records[0].token_ids == (0, 2, 3)
    assert records[0].metadata == {"kind": "mc"}

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"suite": "core"}))
    with pytest.raises(ValueError, match="missing token_ids"):
        builder.load_context_records(bad)

    non_object = tmp_path / "non_object.jsonl"
    non_object.write_text("[]\n")
    with pytest.raises(ValueError, match="JSON object"):
        builder.load_context_records(non_object)

    bad_metadata = tmp_path / "bad_metadata.jsonl"
    bad_metadata.write_text(json.dumps({"token_ids": [0], "metadata": []}) + "\n")
    with pytest.raises(ValueError, match="metadata"):
        builder.load_context_records(bad_metadata)

    null_metadata = tmp_path / "null_metadata.jsonl"
    null_metadata.write_text(json.dumps({"token_ids": [0], "metadata": None}) + "\n")
    assert builder.load_context_records(null_metadata)[0].metadata == {}

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="No context records"):
        builder.load_context_records(empty)


def test_enumerate_contexts_from_config_writes_jsonl_and_parquet(
    tmp_path,
    monkeypatch,
):
    record = EvalContextRecord(
        suite="core",
        task="toy",
        seed=3,
        example_index=4,
        row_index=5,
        token_ids=(0, 99, 100),
        context_text="<bad stale text>",
        metadata={"target_start": 2},
    )

    monkeypatch.setattr(builder, "get_tokenizer", lambda: VisibleBosTokenizer())
    monkeypatch.setattr(
        builder,
        "_enumerate_expected_eval_contexts",
        lambda cfg, tokenizer: [record, record],
    )

    records = builder.enumerate_contexts_from_config(_cfg(tmp_path))

    assert len(records) == 1
    assert records[0].context_text == "ab"
    loaded = builder.load_context_records(tmp_path / "contexts.jsonl")
    assert loaded[0].token_ids == (0, 99, 100)
    assert loaded[0].context_text == "ab"

    table = pq.read_table(tmp_path / "context_parquet" / "shard_00000.parquet")
    assert table.column("text").to_pylist() == ["ab"]
    assert table.column("context_key").to_pylist() == [record.key]
    assert table.column("token_count").to_pylist() == [3]


def test_enumerate_contexts_from_config_rejects_empty_enumeration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder,
        "_enumerate_expected_eval_contexts",
        lambda cfg, tokenizer: [],
    )

    with pytest.raises(ValueError, match="produced no records"):
        builder.enumerate_contexts_from_config(_cfg(tmp_path))


def test_enumerate_validates_feature_clusters_before_tokenizer_or_output(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        stage="enumerate",
        feature_clusters={"path": str(tmp_path / "clusters.yaml")},
    )

    def unexpected_tokenizer():
        raise AssertionError("clustered enumeration reached tokenizer loading")

    monkeypatch.setattr(builder, "get_tokenizer", unexpected_tokenizer)

    with pytest.raises(FeatureClustersError, match="path not found or inaccessible"):
        builder.enumerate_contexts_from_config(cfg)

    assert not (tmp_path / "contexts.jsonl").exists()
    assert not (tmp_path / "context_parquet").exists()


def test_enumerate_expected_eval_contexts_wrapper(monkeypatch):
    import scripts.eval.validate_eval_context_cache as validator

    calls = []
    monkeypatch.setattr(
        validator,
        "enumerate_expected_eval_contexts",
        lambda cfg, tokenizer: calls.append((cfg.stage, tokenizer)) or ["record"],
    )

    result = builder._enumerate_expected_eval_contexts(
        OmegaConf.create({"stage": "enumerate"}),
        "tok",
    )

    assert result == ["record"]
    assert calls == [("enumerate", "tok")]


def test_build_cache_from_config_uses_hydra_missing_values(tmp_path):
    cfg = _cfg(tmp_path, contexts_file="???")

    with pytest.raises(MissingMandatoryValue, match="contexts_file"):
        builder.build_cache_from_config(cfg)


def test_build_cache_from_config_requires_feature_shards(tmp_path):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    (tmp_path / "features").mkdir()

    with pytest.raises(FileNotFoundError, match="No feature shards"):
        builder.build_cache_from_config(_cfg(tmp_path))


def test_build_cache_from_config_rejects_non_mapping_config_blocks(
    tmp_path,
    monkeypatch,
):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(per_feature_stats={}),
    )

    with pytest.raises(TypeError, match="feature_selection"):
        builder.build_cache_from_config(_cfg(tmp_path, feature_selection=["bad"]))

    with pytest.raises(TypeError, match="negative_selection"):
        builder.build_cache_from_config(_cfg(tmp_path, negative_selection=["bad"]))

    # Bypass the helper to test downstream metadata validation: the helper
    # is now imported via a name in steering.conditional.feature_pipeline,
    # so we patch the re-exported reference inside the build_eval_context_cache
    # module.
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        lambda *args, **kwargs: SimpleNamespace(feature_mask=None),
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        lambda **kwargs: [ContextCacheEntry.from_sequences([0, 2, 3], [])],
    )
    with pytest.raises(TypeError, match="metadata"):
        builder.build_cache_from_config(_cfg(tmp_path, metadata=["bad"]))


def test_build_cache_from_config_wires_conversion_and_writes_manifest(
    tmp_path,
    monkeypatch,
):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    calls = {}

    def fake_build_pipeline(cfg_block, **kwargs):
        # Capture exactly what the script passes to the shared helper.
        calls["cfg_block"] = cfg_block
        calls["kwargs"] = kwargs
        return SimpleNamespace(feature_mask=None)

    def fake_build_entries(**kwargs):
        calls["build_entries"] = kwargs
        return [ContextCacheEntry.from_sequences([0, 2, 3], [])]

    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(per_feature_stats={"0": object()}),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        fake_build_pipeline,
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        fake_build_entries,
    )

    cfg = _cfg(
        tmp_path,
        negative_selection={"distribution": "geometric", "mean": 1.0, "std": 1.0},
    )
    manifest = builder.build_cache_from_config(cfg)

    assert manifest == tmp_path / "cache" / "c1024" / ("0" * 16) / "manifest.json"
    assert manifest.is_file()
    provider = SafetensorsContextFeatureProvider(
        tmp_path / "cache" / "c1024" / ("0" * 16),
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=TEST_CACHE_NUM_BUCKETS,
        expected_metadata={"suite": "unit", "model_max_seq_len": 1024},
        identity=TEST_CACHE_IDENTITY,
    )
    assert len(provider) == 1
    # The whole cfg flows through to the helper as the cfg_block argument.
    assert calls["cfg_block"] is cfg
    assert calls["kwargs"]["n_sae_features"] == 8
    assert calls["kwargs"]["num_buckets"] == 4
    # D-seed U-invariant: this cfg has no feature_clusters, so the seed=
    # kwarg must be entirely ABSENT (not seed=None) -- byte-identical to
    # today for every unclustered build regardless of what cfg.seed is set
    # to (this cfg never even defines a top-level 'seed' field).
    assert "seed" not in calls["kwargs"]
    assert calls["build_entries"]["context_records"][0].token_ids == (0, 2, 3)
    assert calls["build_entries"]["feature_shard_paths"] == [feature_path]
    # The pipeline object built by the helper flows straight through.
    assert calls["build_entries"]["pipeline"] is not None
    assert calls["build_entries"]["source_bos_char_len"] == 0


def test_build_cache_from_config_wires_clustered_sampling_and_metadata(
    tmp_path,
    monkeypatch,
):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3]]}),
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(stats_path), metadata={"k": "v"})
    stats_sha = content_sha256(stats_path)
    rng = t.Generator(device="cpu")
    rng.manual_seed(42)
    pipeline = SimpleNamespace(
        feature_mask=None,
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
            ),
            percentile_table=t.zeros(8, 99, dtype=t.float64),
            rng=rng,
        ),
    )
    calls = {}
    pipeline_calls: dict[str, object] = {}

    def fake_build_pipeline(cfg_block, **kwargs):
        # D-seed: a kwargs-recording spy (not a blind lambda that discards
        # everything) so this test can actually assert the real configured
        # seed is threaded into build_feature_pipeline_from_cfg_block for a
        # clustered build. A blind `lambda *a, **k: pipeline` here gave ZERO
        # seed-regression protection -- it would still pass even if the
        # production code silently dropped seed= entirely.
        pipeline_calls["cfg_block"] = cfg_block
        pipeline_calls["kwargs"] = kwargs
        return pipeline

    monkeypatch.setattr(builder, "glob_features", lambda _feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda _stats_path: fixed_statistics_double(per_feature_stats={}),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        fake_build_pipeline,
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())

    def fake_build_entries(**kwargs):
        calls.update(kwargs)
        return [
            ContextCacheEntry.from_sequences(
                [0, 2, 3],
                [ContextFeatureSegment((1,), (2,), 0, 1)],
            )
        ]

    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        fake_build_entries,
    )
    cfg = _cfg(
        tmp_path,
        stats_path=str(stats_path),
        stats_source_sha256=stats_sha,
        feature_clusters={"path": str(cluster_path)},
        feature_selection={
            "distribution": "geometric",
            "mean": 1.0,
            "std": 1.0,
            "min_k": 1,
            "max_k": 2,
        },
        negative_selection=None,
        seed=7,
        identity={
            "model_id": "test-model",
            "sae_id": "test-sae",
            "layer": 0,
            "pooling": "mean",
            "feature_selection_hash": None,
        },
    )

    output_manifest = builder.build_cache_from_config(cfg)

    # D-seed: the real configured seed (not build_feature_pipeline's
    # seed=42 default) must reach the pipeline builder for a clustered
    # build -- this is exactly the kwarg the blind lambda this spy replaced
    # could not observe.
    assert pipeline_calls["kwargs"]["seed"] == 7  # ty: ignore[not-subscriptable]

    resolved = resolve_feature_clusters(cfg, 8)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    vocab = effective_feature_vocab(clusters, 8)
    assert vocab is not None
    manifest = json.loads(output_manifest.read_text())
    assert manifest["metadata"][CLUSTER_SELECTION_SEMANTICS_KEY] == (
        CLUSTER_SELECTION_SEMANTICS
    )
    assert manifest["metadata"]["feature_clusters_canonical"] == (
        resolved.canonical_dict()
    )
    assert manifest["metadata"]["feature_vocab"] == vocab.metadata_dict()
    assert calls["feature_clusters"] == clusters

    provider = SafetensorsContextFeatureProvider(
        output_manifest.parent,
        identity=builder._identity_from_config(cfg),
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=TEST_CACHE_NUM_BUCKETS,
        expected_metadata=manifest["metadata"],
        expected_feature_vocab=vocab,
    )
    assert provider.get([0, 2, 3]) is not None


def _seed_differential_stats(n_features: int = 8) -> dict[str, PerFeatureStat]:
    """Real per-feature percentile stats for 8 features, distinct scales."""
    out: dict[str, PerFeatureStat] = {}
    for feature_id in range(n_features):
        scale = 0.01 * (feature_id + 1)
        percentiles = {f"p{p}": p * scale for p in range(1, 100)}
        out[str(feature_id)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles=percentiles,
        )
    return out


def test_clustered_cache_seed_differential_end_to_end(tmp_path, monkeypatch):
    """D-seed E2E (§7 test design item 4): two REAL builds through the
    actual production ``build_cache_from_config`` entrypoint, identical
    except ``cfg.seed``, must produce (a) a DIFFERENT
    ``identity.feature_selection_hash`` AND (b) DIFFERENT selected content
    for >=1 segment. Deliberately exercises both hunks together (identity +
    content) through the real call chain, with a REAL feature shard (not a
    mocked ``build_feature_pipeline_from_cfg_block``/
    ``build_cache_entries_from_extracted_feature_shards``) -- a
    unit-level shortcut on either hunk alone cannot prove the two actually
    agree end to end, which is the entire point of shipping them as one PR
    (§3: identity alone would be silently, strictly worse than today).
    """
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)

    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3], [2, 4]]}),
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(stats_path), metadata={"k": "v"})
    stats_sha = content_sha256(stats_path)

    # One doc (the standard _write_contexts fixture: BOS + 2 content
    # tokens), one segment spanning the full 2-char decoded content, 8
    # candidate features with distinct magnitudes -- enough spread for a
    # geometric SamplingSelection draw to plausibly differ across seeds.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    shard_data = {
        "offsets": t.tensor([0, 8], dtype=t.int64),
        "feature_ids": t.arange(8, dtype=t.int32),
        "feature_mags": t.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=t.float32),
        "doc_ids": t.zeros(1, dtype=t.int32),
        "segment_indices": t.zeros(1, dtype=t.int32),
        "segment_positions": t.zeros(1, dtype=t.int32),
        "doc_token_counts": t.tensor([3], dtype=t.int32),
        "segment_char_starts": t.zeros(1, dtype=t.int32),
        "segment_char_ends": t.tensor([2], dtype=t.int32),
    }
    save_features(shard_data, feature_dir / "shard_00000_features")

    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    # Only stats *parsing* is stubbed (the dummy stats_path bytes above are
    # not a real safetensors file; they only need to hash consistently for
    # the stats_source_sha256 cross-check). build_feature_pipeline_from_
    # cfg_block and build_cache_entries_from_extracted_feature_shards run
    # FOR REAL below -- that is the entire point of this test.
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda _stats_path: fixed_statistics_double(per_feature_stats=_seed_differential_stats()),
    )

    def build(seed: int):
        cfg = _cfg(
            tmp_path,
            output_dir=str(tmp_path / "cache"),
            stats_path=str(stats_path),
            stats_source_sha256=stats_sha,
            n_sae_features=TEST_CACHE_N_SAE_FEATURES,
            num_buckets=5,
            feature_clusters={"path": str(cluster_path)},
            feature_selection={
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
                "min_k": 1,
                "max_k": 6,
            },
            negative_selection=None,
            seed=seed,
            identity={
                "model_id": "test-model",
                "sae_id": "test-sae",
                "layer": 0,
                "pooling": "mean",
                "feature_selection_hash": None,
            },
        )
        return cfg, builder.build_cache_from_config(cfg)

    cfg_a, manifest_path_a = build(seed=101)
    cfg_b, manifest_path_b = build(seed=202)

    manifest_a = json.loads(manifest_path_a.read_text())
    manifest_b = json.loads(manifest_path_b.read_text())
    hash_a = manifest_a["identity"]["feature_selection_hash"]
    hash_b = manifest_b["identity"]["feature_selection_hash"]
    assert hash_a != hash_b, (
        "two clustered builds differing ONLY in seed must resolve to "
        "DIFFERENT identity.feature_selection_hash values"
    )
    # Distinct hashes -> distinct cache directories under the shared output
    # root (context_length_cache_dir keys on hash_leaf); confirms
    # D-identity's own claim (§3) that every distinct seed now gets its own
    # path rather than colliding onto one universally-wrong cache.
    assert manifest_path_a.parent != manifest_path_b.parent

    resolved = resolve_feature_clusters(cfg_a, 8)
    assert resolved is not None
    clusters = FeatureClusterConsolidation.build(resolved, None)
    vocab = effective_feature_vocab(clusters, 8)
    assert vocab is not None

    provider_a = SafetensorsContextFeatureProvider(
        manifest_path_a.parent,
        identity=builder._identity_from_config(cfg_a),
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=5,
        expected_metadata=manifest_a["metadata"],
        expected_feature_vocab=vocab,
    )
    provider_b = SafetensorsContextFeatureProvider(
        manifest_path_b.parent,
        identity=builder._identity_from_config(cfg_b),
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=5,
        expected_metadata=manifest_b["metadata"],
        expected_feature_vocab=vocab,
    )
    segments_a = provider_a.get([0, 2, 3])
    segments_b = provider_b.get([0, 2, 3])
    assert segments_a is not None
    assert segments_b is not None
    assert segments_a != segments_b, (
        "two clustered builds differing ONLY in seed must select DIFFERENT "
        "content for at least one segment. Without D-content (only "
        "D-identity), every seed's cache directory would be correctly "
        "distinct but silently filled with IDENTICAL "
        "build_feature_pipeline seed=42 content regardless of the "
        "configured seed -- exactly the '§3 strictly worse than today' "
        "scenario this test proves does NOT happen."
    )


# ============================================================================
# D-contract (PR #739 Track D, Option B) Tier 3: THE mandated centerpiece,
# through the real build_cache_from_config entrypoint. The checkpoint-dir
# portion mirrors test_clustered_cache_seed_contract.py's own
# _write_checkpoint (KEEP-LOCAL: small cross-file test fixtures stay
# duplicated rather than cross-imported in this repo) -- MINIMAL, because
# assert_clustered_seed_matches_checkpoint only ever reads the checkpoint's
# own model/metadata files (no packed rows/selection shard/SAE, which the
# OLD full resolve_cluster_feature_contract needed but this narrow
# seed-only check never touches). The rest of the fixture (mask/clusters/
# stats/real content feature shard/contexts.jsonl) is unchanged: those are
# the BUILD's own cfg fields, independent of the checkpoint, and
# build_cache_from_config's live pipeline still needs them to complete the
# real end-to-end run below.
# ============================================================================


def _write_seeded_checkpoint_for_build_fixture(
    tmp_path, *, seed: int, delta_aware: bool = False
) -> dict:
    root = tmp_path / f"seed_{seed}_{int(delta_aware)}"
    root.mkdir()

    n_sae_features = 16_384
    mask_ids = tuple(range(6))
    mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
    mask_path = root / "accepted_feature_ids.txt"
    mask_path.write_text("".join(f"{feature_id}\n" for feature_id in mask_ids), encoding="utf-8")

    cluster_path = root / "feature_clusters.json"
    definition = FeatureClusterDefinition(clusters=((0, 1), (2, 3)))
    cluster_path.write_text(
        json.dumps(definition.canonical_dict(), sort_keys=True), encoding="utf-8"
    )
    resolved_clusters = ResolvedFeatureClusters(
        source_path=cluster_path.resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=n_sae_features
        ),
    )
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)

    feature_dir = build_feature_dir(
        root / "features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
    feature_dir.mkdir(parents=True)
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags={feature_id: float(feature_id + 1) for feature_id in mask_ids},
        absent=range(len(mask_ids), n_sae_features),
    )
    stats = load_statistics(statistics_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, percentile_checksum)
    stats_source_sha256 = content_sha256(statistics_path)

    # Real content feature shard for the ONE context _write_contexts writes
    # (token_ids=[0, 2, 3]; one doc, one segment spanning the full 2-char
    # decoded content) -- mirrors
    # test_clustered_cache_seed_differential_end_to_end's shard shape,
    # restricted to the mask's own allowed feature IDs (0..5) so the live
    # selection pipeline never needs percentile data for a feature this
    # fixture's sparse statistics file has no entry for.
    save_features(
        {
            "offsets": t.tensor([0, 6], dtype=t.int64),
            "feature_ids": t.tensor([0, 1, 2, 3, 4, 5], dtype=t.int32),
            "feature_mags": t.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=t.float32),
            "doc_ids": t.zeros(1, dtype=t.int32),
            "segment_indices": t.zeros(1, dtype=t.int32),
            "segment_positions": t.zeros(1, dtype=t.int32),
            "doc_token_counts": t.tensor([3], dtype=t.int32),
            "segment_char_starts": t.zeros(1, dtype=t.int32),
            "segment_char_ends": t.tensor([2], dtype=t.int32),
        },
        feature_dir / "shard_00000_features",
    )

    # Minimal checkpoint dir: only a loadable model_{step:06d}.pt vocab
    # state-dict + a meta_{step:06d}.json with user_config.seed.
    # delta_aware=True writes the splice-counterfactual/feature-prediction
    # user_config shape -- the one real clustered production checkpoint's
    # shape -- which assert_clustered_seed_matches_checkpoint accepts (the
    # whole point of Option B) but the OLD full contract's
    # _resolve_selection_identity would reject outright.
    checkpoint_dir = root / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert checkpoint_vocab is not None
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((checkpoint_vocab.size, 2)),
            VOCAB_IDS_KEY: t.tensor(checkpoint_vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
        },
        checkpoint_dir / "model_000007.pt",
    )
    positive_selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    negative_selection = {"distribution": "geometric", "mean": 1.0, "std": 1.0}
    user_config = {
        "feature_selection": positive_selection,
        "negative_selection": negative_selection,
        "num_buckets": 2,
        "seed": seed,
        "feature_delta": None,
        "feature_prediction": None,
    }
    if delta_aware:
        user_config["feature_selection"] = {
            **positive_selection,
            "delta_weights": "percentile_delta",
        }
        user_config["feature_delta"] = {"kind": "splice_counterfactual"}
        user_config["feature_prediction"] = {
            "block": "mlp",
            "head_layers": 0,
            "weight": 0.1,
            "layer": -1,
            "loss_mode": "max_pool",
            "loss_fn": "bce",
        }
    checkpoint_metadata = {
        "model_config": {"n_layer": 2, "n_embd": 8},
        "encoder_config": {"n_embd": 8, "depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": mask.canonical_dict(),
        "feature_vocab": checkpoint_vocab.metadata_dict(),
        "feature_clusters_descriptor": resolved_clusters.descriptor_dict(),
        "feature_clusters_canonical": resolved_clusters.canonical_dict(),
        "user_config": user_config,
    }
    (checkpoint_dir / "meta_000007.json").write_text(
        json.dumps(checkpoint_metadata, sort_keys=True), encoding="utf-8"
    )

    contexts_path = root / "contexts.jsonl"
    _write_contexts(contexts_path)

    return {
        "feature_dir": feature_dir,
        "statistics_path": statistics_path,
        "stats_source_sha256": stats_source_sha256,
        "mask_path": mask_path,
        "cluster_path": cluster_path,
        "checkpoint_dir": checkpoint_dir,
        "contexts_path": contexts_path,
        "output_dir": root / "cache",
        "context_parquet_dir": root / "context_parquet",
        "n_sae_features": n_sae_features,
    }


def _cfg_for_seeded_build(tmp_path, fixture: dict, *, seed: int):
    return _cfg(
        tmp_path,
        contexts_file=str(fixture["contexts_path"]),
        context_parquet_dir=str(fixture["context_parquet_dir"]),
        feature_dir=str(fixture["feature_dir"]),
        output_dir=str(fixture["output_dir"]),
        n_sae_features=fixture["n_sae_features"],
        num_buckets=5,
        stats_path=str(fixture["statistics_path"]),
        stats_source_sha256=fixture["stats_source_sha256"],
        feature_selection={
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 3,
        },
        negative_selection=None,
        feature_mask={"mode": "whitelist", "path": str(fixture["mask_path"])},
        feature_clusters={"path": str(fixture["cluster_path"])},
        seed=seed,
        checkpoint_dir=str(fixture["checkpoint_dir"]),
        checkpoint_step=7,
        checkpoint_tag="fixture-step-7",
        identity={
            "model_id": "test-model",
            "sae_id": "test-sae",
            "layer": 0,
            "pooling": "mean",
            "feature_selection_hash": None,
        },
    )


class _GlobFeaturesSpy:
    """Stand-in for steering.feature_io.glob_features that counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _feature_dir):
        self.calls += 1
        return []


class _CallCountingSpy:
    """Counts calls and returns a fixed value -- used to prove a build step
    was NOT reached (e.g. that the seed check raised before it)."""

    def __init__(self, return_value) -> None:
        self.calls = 0
        self._return_value = return_value

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._return_value


def test_build_cache_from_config_seed_mismatch_raises_before_glob_features(tmp_path, monkeypatch):
    """THE binding D-contract centerpiece (plan §1.5/§6 Tier 3 item 7):
    proves the seed cross-check fires strictly BEFORE glob_features (and
    therefore before every other I/O step glob_features precedes) inside
    the real build_cache_from_config entrypoint, not merely somewhere
    before the final manifest write. The checkpoint's REAL preselection
    seed is 99; cfg.seed is deliberately 42.
    """

    fixture = _write_seeded_checkpoint_for_build_fixture(tmp_path, seed=99)
    glob_spy = _GlobFeaturesSpy()
    # load_context_records is the EARLIEST build I/O (it opens the contexts
    # JSONL, several lines before glob_features). Spying it -- not merely
    # glob_features -- is what actually pins "before ANY shard/context I/O":
    # a regression moving the seed check between load_context_records and
    # glob_features would leave glob_spy at 0 yet trip records_spy, so the
    # weaker glob-only assertion could not catch it.
    records_spy = _CallCountingSpy(return_value=[])
    monkeypatch.setattr(builder, "glob_features", glob_spy)
    monkeypatch.setattr(builder, "load_context_records", records_spy)
    cfg = _cfg_for_seeded_build(tmp_path, fixture, seed=42)

    with pytest.raises(ClusterFeatureContractError) as exc_info:
        builder.build_cache_from_config(cfg)

    message = str(exc_info.value)
    assert "42" in message
    assert "99" in message
    assert records_spy.calls == 0, (
        "assert_clustered_seed_matches_checkpoint must raise strictly BEFORE "
        "load_context_records (the earliest build I/O) -- a nonzero count "
        "means the seed check ran after context I/O had already started."
    )
    assert glob_spy.calls == 0, (
        "the seed check must also precede glob_features (already implied by "
        "the load_context_records guard above; asserted explicitly)."
    )


def test_build_cache_from_config_seed_match_completes_and_stamps_checkpoint_metadata(
    tmp_path, monkeypatch
):
    """D-contract acceptance: the matching-seed companion to the centerpiece
    above -- a real build_cache_from_config run all the way through, with
    the manifest metadata carrying all 5 checkpoint_*-prefixed provenance
    keys ``CheckpointSeedProvenance`` resolved. ``checkpoint_selection_hash``
    is GONE (Track B's full selection contract is no longer run) -- replaced
    by ``checkpoint_resolved_tag`` (see
    ``clustered_cache_seed_contract.CheckpointSeedProvenance.metadata_dict``).
    """

    fixture = _write_seeded_checkpoint_for_build_fixture(tmp_path, seed=99)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    cfg = _cfg_for_seeded_build(tmp_path, fixture, seed=99)

    manifest_path = builder.build_cache_from_config(cfg)

    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["metadata"]
    expected_keys = {
        "checkpoint_resolved_tag",
        "checkpoint_model_sha256",
        "checkpoint_metadata_sha256",
        "checkpoint_step",
        "checkpoint_preselection_seed",
    }
    assert expected_keys <= set(metadata.keys())
    assert "checkpoint_selection_hash" not in metadata
    assert metadata["checkpoint_preselection_seed"] == 99
    assert metadata["checkpoint_step"] == 7
    assert metadata["checkpoint_resolved_tag"] == "fixture-step-7"
    for key in ("checkpoint_model_sha256", "checkpoint_metadata_sha256"):
        digest = metadata[key]
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


def test_build_cache_from_config_delta_aware_checkpoint_seed_match_completes(
    tmp_path, monkeypatch
):
    """D-contract Option B, end to end: a delta-aware (splice-counterfactual
    + feature-prediction) checkpoint -- the exact shape the OLD full
    ``resolve_cluster_feature_contract`` rejected via
    ``_resolve_selection_identity``'s ``delta_weights``/``feature_delta``/
    ``feature_prediction`` guards (see
    ``test_clustered_cache_seed_contract.py::
    test_delta_aware_checkpoint_would_fail_full_contract``) -- now completes
    a full ``build_cache_from_config`` run when its seed matches, proving
    Option B's acceptance through the real production entrypoint, not just
    the narrow ``assert_clustered_seed_matches_checkpoint`` unit tests.
    """

    fixture = _write_seeded_checkpoint_for_build_fixture(tmp_path, seed=99, delta_aware=True)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    cfg = _cfg_for_seeded_build(tmp_path, fixture, seed=99)

    manifest_path = builder.build_cache_from_config(cfg)

    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["metadata"]
    assert metadata["checkpoint_preselection_seed"] == 99
    assert metadata["checkpoint_step"] == 7


def test_build_cache_requires_exact_stage3e_contract_and_derives_bos_length(
    tmp_path,
    monkeypatch,
):
    manifest_path, jsonl_path, parquet_path, stage3e_manifest = (
        _write_required_stage3e_contract(tmp_path)
    )
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    calls = {}

    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda tokenizer_id: (
            VisibleBosTokenizer()
            if tokenizer_id == "test/extraction-tokenizer"
            else pytest.fail(f"unexpected tokenizer id: {tokenizer_id}")
        ),
    )
    monkeypatch.setattr(builder, "glob_features", lambda _feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda _stats_path: fixed_statistics_double(per_feature_stats={}),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        lambda *args, **kwargs: SimpleNamespace(feature_mask=None),
    )

    def fake_build_entries(**kwargs):
        calls.update(kwargs)
        return [ContextCacheEntry.from_sequences([0, 99, 100], [])]

    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        fake_build_entries,
    )
    cfg = _cfg(
        tmp_path,
        contexts_file=str(jsonl_path),
        context_parquet_dir=str(parquet_path.parent),
        required_contexts_manifest=str(manifest_path),
        extraction_tokenizer_id="test/extraction-tokenizer",
        source_bos_char_len=None,
    )

    output_manifest = builder.build_cache_from_config(cfg)

    assert calls["source_bos_char_len"] == len(
        VisibleBosTokenizer().decode([VisibleBosTokenizer().get_bos_token_id()])
    )
    extraction_checksum = compute_tokenizer_checksum(VisibleBosTokenizer())
    provider = SafetensorsContextFeatureProvider(
        output_manifest.parent,
        expected_n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        expected_num_buckets=TEST_CACHE_NUM_BUCKETS,
        expected_metadata={
            "suite": "unit",
            "model_max_seq_len": 1024,
            "contexts_fingerprint": stage3e_manifest["contexts_fingerprint"],
            "context_extraction_tokenizer_checksum": extraction_checksum,
        },
        identity=TEST_CACHE_IDENTITY,
    )
    assert len(provider) == 1


def test_build_cache_rejects_contexts_not_committed_by_stage3e(
    tmp_path,
    monkeypatch,
):
    manifest_path, _jsonl_path, parquet_path, _manifest = (
        _write_required_stage3e_contract(tmp_path)
    )
    stale_jsonl = tmp_path / "stale_contexts.jsonl"
    _write_contexts(stale_jsonl)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())

    with pytest.raises(ValueError, match="does not match.*JSONL"):
        builder.build_cache_from_config(
            _cfg(
                tmp_path,
                contexts_file=str(stale_jsonl),
                context_parquet_dir=str(parquet_path.parent),
                required_contexts_manifest=str(manifest_path),
                extraction_tokenizer_id="unused",
                source_bos_char_len=None,
            )
        )


def test_build_cache_rejects_parquet_not_committed_by_stage3e(
    tmp_path,
    monkeypatch,
):
    manifest_path, jsonl_path, parquet_path, _manifest = (
        _write_required_stage3e_contract(tmp_path)
    )
    stale_dir = tmp_path / "stale_parquet"
    stale_dir.mkdir()
    stale_parquet = stale_dir / parquet_path.name
    shutil.copyfile(parquet_path, stale_parquet)
    stale_parquet.write_bytes(stale_parquet.read_bytes() + b"stale")
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())

    with pytest.raises(ValueError, match="exact parquet"):
        builder.build_cache_from_config(
            _cfg(
                tmp_path,
                contexts_file=str(jsonl_path),
                context_parquet_dir=str(stale_dir),
                required_contexts_manifest=str(manifest_path),
                extraction_tokenizer_id="unused",
                source_bos_char_len=None,
            )
        )


def test_build_cache_rejects_supplied_bos_length_disagreement(
    tmp_path,
    monkeypatch,
):
    manifest_path, jsonl_path, parquet_path, _manifest = (
        _write_required_stage3e_contract(tmp_path)
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder.AutoTokenizer,
        "from_pretrained",
        lambda _tokenizer_id: VisibleBosTokenizer(),
    )

    with pytest.raises(ValueError, match="source_bos_char_len disagrees"):
        builder.build_cache_from_config(
            _cfg(
                tmp_path,
                contexts_file=str(jsonl_path),
                context_parquet_dir=str(parquet_path.parent),
                required_contexts_manifest=str(manifest_path),
                extraction_tokenizer_id="test/extraction-tokenizer",
                source_bos_char_len=0,
            )
        )


def test_build_cache_from_config_separates_output_by_context_length(
    tmp_path,
    monkeypatch,
):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()

    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(per_feature_stats={}),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        lambda *args, **kwargs: SimpleNamespace(feature_mask=None),
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        lambda **kwargs: [ContextCacheEntry.from_sequences([0, 2, 3], [])],
    )

    root = tmp_path / "cache"
    manifest_1024 = builder.build_cache_from_config(
        _cfg(tmp_path, output_dir=str(root), model_max_seq_len=1024)
    )
    manifest_2048 = builder.build_cache_from_config(
        _cfg(tmp_path, output_dir=str(root), model_max_seq_len=2048)
    )

    assert manifest_1024 == root / "c1024" / ("0" * 16) / "manifest.json"
    assert manifest_2048 == root / "c2048" / ("0" * 16) / "manifest.json"
    assert (
        SafetensorsContextFeatureProvider(
            root / "c1024" / ("0" * 16),
            expected_metadata={"model_max_seq_len": 1024},
            identity=TEST_CACHE_IDENTITY,
        ).get([0, 2, 3])
        == ()
    )
    assert (
        SafetensorsContextFeatureProvider(
            root / "c2048" / ("0" * 16),
            expected_metadata={"model_max_seq_len": 2048},
            identity=TEST_CACHE_IDENTITY,
        ).get([0, 2, 3])
        == ()
    )


def test_build_cache_from_config_rejects_metadata_context_len_mismatch(
    tmp_path,
    monkeypatch,
):
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(per_feature_stats={}),
    )
    monkeypatch.setattr(
        builder,
        "build_feature_pipeline_from_cfg_block",
        lambda *args, **kwargs: SimpleNamespace(feature_mask=None),
    )
    monkeypatch.setattr(builder, "get_tokenizer", lambda: MockTokenizer())
    monkeypatch.setattr(
        builder,
        "build_cache_entries_from_extracted_feature_shards",
        lambda **kwargs: [ContextCacheEntry.from_sequences([0, 2, 3], [])],
    )

    with pytest.raises(ValueError, match="metadata.model_max_seq_len"):
        builder.build_cache_from_config(_cfg(tmp_path, metadata={"model_max_seq_len": 2048}))


def test_validate_cache_from_config_rejects_missing_and_empty_output(tmp_path):
    with pytest.raises(MissingMandatoryValue, match="output_dir"):
        builder.validate_cache_from_config(
            OmegaConf.create(
                {
                    "output_dir": "???",
                    "model_max_seq_len": 1024,
                    "n_sae_features": 8,
                    "num_buckets": 4,
                }
            )
        )

    empty_cache = context_length_cache_dir(
        tmp_path / "empty_cache",
        1024,
        hash_leaf="0" * 16,
    )
    save_context_cache(
        empty_cache,
        [],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"model_max_seq_len": 1024},
        identity=TEST_CACHE_IDENTITY,
    )
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "empty_cache"),
            "model_max_seq_len": 1024,
            "n_sae_features": 8,
            "num_buckets": 4,
            "identity": {
                "model_id": "test-model",
                "sae_id": "test-sae",
                "layer": 0,
                "pooling": "mean",
                "feature_selection_hash": "0" * 64,
            },
        }
    )
    with pytest.raises(ValueError, match="empty"):
        builder.validate_cache_from_config(cfg)


def test_validate_resolves_feature_clusters_before_identity_or_provider(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(
        tmp_path,
        stage="validate",
        feature_clusters={"path": str(tmp_path / "clusters.yaml")},
    )

    def unexpected_identity(_cfg):
        raise AssertionError("clustered validation reached cache identity")

    monkeypatch.setattr(builder, "_identity_from_config", unexpected_identity)

    with pytest.raises(FeatureClustersError, match="path not found or inaccessible"):
        builder.validate_cache_from_config(cfg)


def test_validate_cache_rejects_removed_clustering_before_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _cfg(
        tmp_path,
        stage="validate",
        stats_source_sha256="a" * 64,
        feature_clustering={"enabled": False},
        identity={
            "model_id": "test-model",
            "sae_id": "test-sae",
            "layer": 0,
            "pooling": "mean",
            "feature_selection_hash": None,
        },
    )

    def fail_if_provider_constructed(*_args, **_kwargs):
        raise AssertionError("stale feature_clustering reached cache provider")

    monkeypatch.setattr(
        builder,
        "SafetensorsContextFeatureProvider",
        fail_if_provider_constructed,
    )

    with pytest.raises(RemovedFeatureError, match="eval context cache validation"):
        builder.validate_cache_from_config(cfg)


def test_validate_cache_from_config_accepts_nonempty_cache(tmp_path):
    cache = context_length_cache_dir(
        tmp_path / "cache",
        1024,
        hash_leaf="0" * 16,
    )
    save_context_cache(
        cache,
        [ContextCacheEntry.from_sequences([0, 2, 3], [])],
        n_sae_features=TEST_CACHE_N_SAE_FEATURES,
        num_buckets=TEST_CACHE_NUM_BUCKETS,
        metadata={"model_max_seq_len": 1024},
        identity=TEST_CACHE_IDENTITY,
    )
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "cache"),
            "model_max_seq_len": 1024,
            "n_sae_features": 8,
            "num_buckets": 4,
            "identity": {
                "model_id": "test-model",
                "sae_id": "test-sae",
                "layer": 0,
                "pooling": "mean",
                "feature_selection_hash": "0" * 64,
            },
        }
    )

    builder.validate_cache_from_config(cfg)


def test_main_dispatches_stages(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        builder,
        "enumerate_contexts_from_config",
        lambda cfg: calls.append("enumerate"),
    )
    monkeypatch.setattr(builder, "build_cache_from_config", lambda cfg: calls.append("convert"))
    monkeypatch.setattr(
        builder,
        "validate_cache_from_config",
        lambda cfg: calls.append("validate"),
    )
    entrypoint = builder.main.__wrapped__

    entrypoint(OmegaConf.create({"stage": "enumerate"}))
    entrypoint(OmegaConf.create({"stage": "convert"}))
    entrypoint(OmegaConf.create({"stage": "validate"}))
    with pytest.raises(ValueError):
        entrypoint(OmegaConf.create({"stage": "other"}))

    assert calls == ["enumerate", "convert", "validate"]


def test_build_cache_from_config_refuses_gradient_ranking(tmp_path, monkeypatch):
    """Context-cache building live-selects per segment: a gradient-ranked
    feature_selection must fail fast HERE (before any conversion work), not
    at the first sample_features call. Regression-pins the guard's wiring —
    the tolerant getattr for stubbed pipelines must not neuter it on real
    pipelines."""
    from errors import GradientRankingUnsupportedError

    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts)
    feature_path = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    feature_path.parent.mkdir()
    monkeypatch.setattr(builder, "glob_features", lambda feature_dir: [feature_path])
    monkeypatch.setattr(
        builder,
        "load_statistics",
        lambda stats_path: fixed_statistics_double(per_feature_stats={}),
    )
    cfg = _cfg(
        tmp_path,
        feature_selection={
            "distribution": "geometric",
            "mean": 5.0,
            "mode": "top_k_percentile",
            "gradient": {"mode": "future_half_split"},
        },
    )
    with pytest.raises(GradientRankingUnsupportedError, match="offline-only"):
        builder.build_cache_from_config(cfg)
