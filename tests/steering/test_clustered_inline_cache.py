"""Clustered inline-steering cache integration and identity tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.eval.clustered_inline_cache as clustered_cache_module
import steering.eval.inline_steering_eval as inline_eval_module
from errors import FeatureClustersError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import resolve_feature_mask
from steering.eval.clustered_inline_cache import (
    CLUSTERED_INLINE_CACHE_SEMANTICS,
    ClusteredInlineEvalPayload,
    assert_clustered_inline_cache_ddp_consistent,
    prepare_clustered_inline_eval_cache,
)
from steering.eval.inline_steering_eval import (
    InlineEvalCache,
    SteeringEvalConfig,
    evaluate_steering,
)
from steering.overlay_io import read_manifest
from tests.steering._splice_cf_cache_fixtures import (
    SELECTION_N_SAE,
    build_interleaved_env,
    run_selection_main,
    write_cluster_yaml,
)

_WHITELIST_MASK = {"mode": "whitelist", "ids": list(range(200))}


def _gloo_cache_digest_target(
    rank: int,
    world_size: int,
    disagree: bool,
) -> str:
    digest = ("b" if disagree and rank == 1 else "a") * 64
    payload = ClusteredInlineEvalPayload(
        source_identity={},
        targets=(),
        digest=digest,
    )
    try:
        assert_clustered_inline_cache_ddp_consistent(
            payload,
            world_size=world_size,
        )
    except RuntimeError as exc:
        return str(exc)
    return "ok"


def _gloo_clustered_eval_failure_target(rank: int, world_size: int) -> str:
    payload = ClusteredInlineEvalPayload(
        source_identity={},
        targets=(),
        digest="a" * 64,
    )
    cache = InlineEvalCache(
        selected_docs=[(0, 1), (1, 1)],
        selection_metrics={
            "steering/n_docs_available": 2,
            "steering/n_docs_selected": 2,
        },
        doc_texts={0: "zero", 1: "one"},
        clustered_payload=payload,
    )

    def generate_or_fail(*_args, **_kwargs):
        if rank == 1:
            raise ValueError("rank-local boom")
        return []

    inline_eval_module._generate_clustered_roundtrip_samples = generate_or_fail  # ty: ignore[invalid-assignment]

    class EvalModel:
        training = False

    try:
        evaluate_steering(
            model=EvalModel(),
            feature_dir=Path("."),
            tokenizer=object(),
            device=t.device("cpu"),
            boundaries=None,
            threshold_cfg=None,
            precomputed_thresholds=None,
            cfg=SteeringEvalConfig(),
            eval_cache=cache,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("clustered DDP evaluation unexpectedly succeeded")


def _cluster_state(
    cluster_path: str,
) -> tuple[object, FeatureClusterConsolidation]:
    cfg = OmegaConf.create(
        {
            "feature_mask": _WHITELIST_MASK,
            "feature_clusters": {"path": cluster_path},
        }
    )
    feature_mask = resolve_feature_mask(cfg, SELECTION_N_SAE)
    resolved = resolve_feature_clusters(cfg, SELECTION_N_SAE)
    assert resolved is not None
    return feature_mask, FeatureClusterConsolidation.build(resolved, feature_mask)


def test_clustered_inline_cache_two_pass_join_is_deterministic_and_snapshot_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env = build_interleaved_env(tmp_path, n_shards=3)
    cluster_path = write_cluster_yaml(tmp_path)
    run_selection_main(
        env,
        mask=_WHITELIST_MASK,
        clusters_path=cluster_path,
        num_workers=1,
        splice_batch_size=4,
        monkeypatch=monkeypatch,
    )

    selection_dirs = sorted((env["packed_row_dir"] / "sel").iterdir())
    assert len(selection_dirs) == 1
    selection_dir = selection_dirs[0]
    manifest = read_manifest(selection_dir)
    assert manifest is not None
    splice_metadata = manifest["metadata"]["splice_counterfactual"]

    feature_mask, clusters = _cluster_state(cluster_path)
    vocab = effective_feature_vocab(clusters, SELECTION_N_SAE)
    assert vocab is not None
    feature_index = FeatureIndex(
        env["feature_dir"],
        percentile_checksum=env["pct_checksum"],
        tokenizer_checksum=env["tok_checksum"],
    )
    # The tiny packer uses only one document from its final source shard, but
    # the middle derived shard contains two positive-selected documents. Point
    # the evaluator's validation corpus at that exact source stem so the
    # different-document donor contract is exercised rather than weakened.
    validation_data_dir = tmp_path / "validation_data"
    validation_data_dir.mkdir()
    shutil.copy2(
        Path(env["data_dir"]) / "shard_00001.parquet",
        validation_data_dir / "shard_00001.parquet",
    )
    kwargs = {
        "feature_dir": Path(env["feature_dir"]),
        "packed_row_dir": env["packed_row_dir"],
        "selection_dir": selection_dir,
        "selection_hash": selection_dir.name,
        "feature_index": feature_index,
        "data_dir": str(validation_data_dir),
        "n_eval_docs": 2,
        "seed": 42,
        "model_max_seq_len": 2_048,
        "effective_buckets": 1,
        "feature_mask": feature_mask,
        "feature_clusters": clusters,
        "feature_vocab": vocab,
        "delta_config": {
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        },
        "boundary_fingerprint": splice_metadata["boundary_fingerprint"],
        "splice_batch_size": 4,
    }

    first = prepare_clustered_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]
    second = prepare_clustered_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]

    assert first.payload.digest == second.payload.digest
    assert first.selected_docs == second.selected_docs
    assert first.payload.source_identity["semantics"] == CLUSTERED_INLINE_CACHE_SEMANTICS
    assert first.payload.source_identity["selection_hash"] == selection_dir.name
    assert first.payload.source_identity["feature_vocab"] == vocab.metadata_dict()
    assert first.payload.source_identity["model_max_seq_len"] == 2_048
    assert first.selected_docs
    assert len(first.payload.targets) == len(first.selected_docs)

    certified = prepare_clustered_inline_eval_cache(
        **{
            **kwargs,
            "validation_shard_stem": "shard_00001",
            "remote_derived_feature_fingerprint": splice_metadata[
                "derived_feature_fingerprint"
            ],
            "source_certificate_fingerprint": "c" * 64,
        }  # ty: ignore[invalid-argument-type]
    )
    assert certified.selected_docs == first.selected_docs
    assert certified.payload.source_identity["remote_inline_eval_source"] == (
        "c" * 64
    )
    assert certified.payload.digest != first.payload.digest

    with pytest.raises(ValueError, match="verified remote startup contracts"):
        prepare_clustered_inline_eval_cache(
            **{
                **kwargs,
                "validation_shard_stem": "shard_00001",
                "remote_derived_feature_fingerprint": "f" * 64,
                "source_certificate_fingerprint": "c" * 64,
            }  # ty: ignore[invalid-argument-type]
        )

    for target in first.payload.targets:
        assert target.donor_doc_idx != target.doc_idx
        assert len(target.prefix_segments) == target.target_segment_ordinal + 1
        assert set(target.positive_feature_ids).isdisjoint(target.negative_feature_ids)
        assert len(target.positive_member_groups) == len(target.positive_feature_ids)
        for representative, members in zip(
            target.positive_feature_ids,
            target.positive_member_groups,
            strict=True,
        ):
            assert representative in members
            assert members == tuple(sorted(set(members)))
        for segment in (*target.prefix_segments, target.donor_segment):
            clusters.assert_representatives_only(
                segment.feature_ids,
                surface="clustered inline cache test",
            )

    longest_selected_end = max(
        int(target.prefix_segments[-1].token_end or 0)
        for target in first.payload.targets
    )
    context_limited = prepare_clustered_inline_eval_cache(
        **{
            **kwargs,
            "model_max_seq_len": longest_selected_end - 1,
        }  # ty: ignore[invalid-argument-type]
    )
    assert (
        context_limited.selection_metrics[
            "steering/n_targets_context_filtered"
        ]
        > 0
    )
    assert all(
        int(target.prefix_segments[-1].token_end or 0) < longest_selected_end
        for target in context_limited.payload.targets
    )
    assert context_limited.payload.digest != first.payload.digest

    real_validate = clustered_cache_module.validate_complete_selection_dir
    validation_calls = 0

    def validate_then_replace_manifest(**call_kwargs):
        nonlocal validation_calls
        validated = real_validate(**call_kwargs)
        validation_calls += 1
        if validation_calls == 2:
            manifest_path = selection_dir / "manifest.json"
            replacement = json.loads(manifest_path.read_text(encoding="utf-8"))
            replacement["metadata"]["replaced_after_final_validation"] = True
            manifest_path.write_text(
                json.dumps(replacement, sort_keys=True),
                encoding="utf-8",
            )
        return validated

    monkeypatch.setattr(
        clustered_cache_module,
        "validate_complete_selection_dir",
        validate_then_replace_manifest,
    )
    with pytest.raises(ValueError, match="manifest identity changed"):
        prepare_clustered_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]


def test_clustered_inline_cache_requires_splice_counterfactual(
    tmp_path: Path,
) -> None:
    cluster_path = write_cluster_yaml(tmp_path)
    feature_mask, clusters = _cluster_state(cluster_path)
    vocab = effective_feature_vocab(clusters, SELECTION_N_SAE)
    assert vocab is not None

    with pytest.raises(FeatureClustersError, match="requires.*splice_counterfactual"):
        prepare_clustered_inline_eval_cache(
            feature_dir=tmp_path,
            packed_row_dir=tmp_path,
            selection_dir=tmp_path,
            selection_hash="0" * 64,
            feature_index=object(),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            data_dir=None,
            n_eval_docs=1,
            seed=42,
            model_max_seq_len=2_048,
            effective_buckets=1,
            feature_mask=feature_mask,  # ty: ignore[invalid-argument-type]
            feature_clusters=clusters,
            feature_vocab=vocab,
            delta_config=None,
            boundary_fingerprint=None,
        )


def test_clustered_inline_cache_digest_disagreement_fails(
    monkeypatch,
) -> None:
    class _Payload:
        digest = "a" * 64

    monkeypatch.setattr(
        "steering.eval.clustered_inline_cache.dist.is_initialized",
        lambda: True,
    )

    def fake_all_gather(output, value):
        output[:] = [value, "b" * 64]

    monkeypatch.setattr(
        "steering.eval.clustered_inline_cache.dist.all_gather_object",
        fake_all_gather,
    )
    with pytest.raises(RuntimeError, match="digest disagreement"):
        assert_clustered_inline_cache_ddp_consistent(
            _Payload(),  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            world_size=2,
        )


@pytest.mark.parametrize("disagree", [False, True])
def test_clustered_inline_cache_digest_uses_real_two_rank_gloo(
    gloo_multiprocess,
    disagree: bool,
) -> None:
    results = gloo_multiprocess(2, _gloo_cache_digest_target, disagree)
    if disagree:
        assert results[0] == results[1]
        assert "digest disagreement" in results[0]
        assert "aaaaaaaa" in results[0]
        assert "bbbbbbbb" in results[0]
    else:
        assert results == ["ok", "ok"]


def test_clustered_eval_failure_is_symmetric_under_real_two_rank_gloo(
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(2, _gloo_clustered_eval_failure_target)
    assert results[0] == results[1]
    assert "rank 1: ValueError: rank-local boom" in results[0]
