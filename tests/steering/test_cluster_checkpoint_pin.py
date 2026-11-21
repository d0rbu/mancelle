"""Tests for clustered-evaluation checkpoint and artifact identity checks.

Cover checkpoint directory/step resolution, metadata normalization, hashing
and vocabulary agreement, statistics and SAE compatibility, selection discovery,
and certified/forced validation paths using serialized production-format fixtures.

Exercise invalid inputs, artifact mutations, cache behavior, and comparison with
frozen validator oracles. Accepted stat-preserving and sequential-mutation
schedules are explicit scope tests; see ``cluster_checkpoint_pin``'s module
docstring for the validation boundaries.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from safetensors.torch import load_file, save_file

import steering.conditional.splice_counterfactual as splice_module
import steering.eval.cluster_checkpoint_pin as ccp
import steering.eval.cluster_contract_identity as cci
import steering.stats_io as stats_io_module
from errors import FeatureClustersError
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    build_clustered_selection_validation,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    DeltaMode,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    assert_clustered_selection_shard_integrity,
    build_clustered_selection_shard_integrity,
    canonical_feature_clusters_descriptor,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_mask import canonical_dict as canonical_feature_mask_dict
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.splice_counterfactual import (
    packed_row_fingerprint,
    resolve_packed_row_fingerprint_strict,
)
from steering.conditional.splice_selection_certificate import (
    SPLICE_RESUME_METADATA_KEY,
    SpliceSelectionStats,
    build_splice_resume_metadata,
    parse_splice_resume_metadata,
)
from steering.eval.cluster_checkpoint_pin import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContractError,
    resolve_checkpoint_pin,
    resolve_eval_checkpoint_directory,
    resolve_selection_manifest_path,
)
from steering.eval.cluster_config import (
    CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    CLUSTERED_ENCDEC_TARGET_METHODS,
    CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.immutable_io import is_sha256_hex
from steering.overlay_io import OverlayKind, read_manifest, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    compute_selection_hash,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    build_remote_loader_certificate,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.steering._splice_cf_cache_fixtures import (
    WHITELIST_MASK_BELOW_150,
    build_interleaved_env,
    run_selection_main,
    write_cluster_yaml,
)

_A_SHA = "a" * 64
_B_SHA = "b" * 64
_C_SHA = "c" * 64
_D_SHA = "d" * 64


# --------------------------------------------------------------------------
# Checkpoint, cluster, and metadata fixtures built without a full feature contract.
# --------------------------------------------------------------------------


def _clusters(
    source_path: Path,
    *,
    clusters: tuple[tuple[int, ...], ...] = ((1, 2), (4, 5)),
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=clusters)
    return ResolvedFeatureClusters(
        source_path=source_path,
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition,
            n_sae_features=8,
        ),
    )


def _pin(
    root: Path,
    mask: FeatureMask | None,
    *,
    model_sha: str = _C_SHA,
    feature_clusters: ResolvedFeatureClusters | None = None,
) -> CheckpointPin:
    """Build a pin with the checkpoint's effective feature vocabulary.

    Use mask-only compaction without clusters; otherwise derive the vocabulary
    from the mask and cluster consolidation together.
    """
    digest = cci.StableFileDigest(
        path=root / "model_000007.pt",
        sha256=model_sha,
        byte_length=123,
        stat_signature=(0o100644, 1, 2, 123, 4, 5),
    )
    metadata_digest = cci.StableFileDigest(
        path=root / "meta_000007.json",
        sha256=_D_SHA,
        byte_length=45,
        stat_signature=(0o100644, 1, 3, 45, 4, 5),
    )
    if feature_clusters is None:
        feature_vocab = FeatureVocab.from_mask(mask, 8)
    else:
        feature_vocab = effective_feature_vocab(
            FeatureClusterConsolidation.build(feature_clusters, mask), 8
        )
    return CheckpointPin(
        checkpoint_dir=root,
        resolved_tag="clustered-d12",
        step=7,
        model_file=digest,
        metadata_file=metadata_digest,
        n_sae_features=8,
        feature_vocab=feature_vocab,
        load_bearing_metadata_json="{}",
    )


def _write_checkpoint(
    directory: Path,
    *,
    step: int,
    ids: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    n_sae_features: int = 8,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((len(ids), 2)),
            VOCAB_IDS_KEY: t.tensor(ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
        },
        directory / f"model_{step:06d}.pt",
    )
    (directory / f"meta_{step:06d}.json").write_text(
        json.dumps({"user_config": {}}),
        encoding="utf-8",
    )


def _agreement_metadata(
    feature_mask: FeatureMask | None,
    checkpoint: CheckpointPin,
    feature_clusters: ResolvedFeatureClusters | None,
) -> dict[str, Any]:
    return {
        "feature_mask_canonical": canonical_feature_mask_dict(feature_mask),
        "feature_vocab": (
            None if checkpoint.feature_vocab is None else checkpoint.feature_vocab.metadata_dict()
        ),
        "feature_clusters_descriptor": (
            None if feature_clusters is None else feature_clusters.descriptor_dict()
        ),
        "feature_clusters_canonical": (
            None if feature_clusters is None else feature_clusters.canonical_dict()
        ),
        "user_config": {},
    }


def _base_agreement_fixture(
    tmp_path: Path,
) -> tuple[FeatureMask, CheckpointPin, ResolvedFeatureClusters]:
    """Build an agreeing mask, pin, and cluster configuration at SAE width 8.

    Mask ``(1..6)`` and clusters ``((1,2),(4,5))`` fold 2 into 1 and 5 into 4,
    so the checkpoint's effective vocabulary is ``(1, 3, 4, 6)``.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    checkpoint = _pin(tmp_path / "checkpoint", mask, feature_clusters=clusters)
    assert checkpoint.feature_vocab == FeatureVocab(8, (1, 3, 4, 6))
    return mask, checkpoint, clusters


def _write_stats_feature_dir(
    tmp_path: Path,
    *,
    n_sae_features: int = 16_384,
    segment_size: int = 128,
    layer: int = 8,
    pooling: PoolingMode = PoolingMode.max_pool,
    width: SAEWidth = SAEWidth.W_16K,
    # write_steering_stats never sets ExtractionStatistics.ignore_bos_features
    # explicitly, so real stats built by it always load with the dataclass
    # default (False) — the default here must match, or every "happy path"
    # caller would accidentally trip the ignore_bos_features mismatch check.
    ignore_bos_features: bool = False,
    segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    feature_ids: tuple[int, ...] = (0, 1, 2),
    name: str = "features",
) -> tuple[Path, Path]:
    feature_dir = build_feature_dir(
        tmp_path / name,
        segment_size=segment_size,
        layer=layer,
        activation_type=ActivationType.residual,
        pooling=pooling,
        width=width,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=ignore_bos_features,
        segment_mode=segment_mode,
    )
    feature_dir.mkdir(parents=True)
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags={fid: float(fid + 1) for fid in feature_ids},
        absent=[i for i in range(n_sae_features) if i not in feature_ids],
    )
    stats = load_statistics(statistics_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, percentile_checksum)
    return feature_dir, statistics_path


# --------------------------------------------------------------------------
# ``_mapping_get``
# --------------------------------------------------------------------------


def test_mapping_get_reads_dict_and_object_attributes_with_default() -> None:
    assert ccp._mapping_get({"a": 1}, "a") == 1
    assert ccp._mapping_get({"a": 1}, "missing", "default") == "default"

    class _Obj:
        a = 1

    assert ccp._mapping_get(_Obj(), "a") == 1
    assert ccp._mapping_get(_Obj(), "missing", "default") == "default"


# --------------------------------------------------------------------------
# ``_checkpoint_load_bearing_metadata``
# --------------------------------------------------------------------------


def test_checkpoint_load_bearing_metadata_selects_declared_keys_only() -> None:
    metadata = {
        "model_config": {"n_layer": 2},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": {"mode": "whitelist", "feature_ids": [1]},
        "feature_vocab": {"n_sae_features": 8},
        "feature_clusters_descriptor": {"schema_version": 1},
        "feature_clusters_canonical": {"schema_version": 1, "clusters": []},
        "user_config": {
            "seed": 42,
            "sae": {"release": "gemmascope"},
            "unrelated_key": "must not leak",
            "learning_rate": 0.001,
        },
        "top_level_unrelated": "must not leak",
    }
    result = ccp._checkpoint_load_bearing_metadata(metadata)
    assert result == {
        "model_config": {"n_layer": 2},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": {"mode": "whitelist", "feature_ids": [1]},
        "feature_vocab": {"n_sae_features": 8},
        "feature_clusters_descriptor": {"schema_version": 1},
        "feature_clusters_canonical": {"schema_version": 1, "clusters": []},
        "user_config": {"seed": 42, "sae": {"release": "gemmascope"}},
    }


def test_checkpoint_load_bearing_metadata_omits_absent_user_config_keys() -> None:
    result = ccp._checkpoint_load_bearing_metadata({"user_config": {}})
    assert result["user_config"] == {}
    assert result["model_config"] is None


def test_checkpoint_load_bearing_metadata_parity_between_pretrain_and_sft_layouts() -> None:
    """Normalize nested pretrain and top-level SFT metadata identically.

    Full-dict equality catches stray layout-specific fields, such as an extra
    ``user_config.num_buckets``. Independent field checks also ensure that both
    layouts retain the expected values rather than agreeing on wrong/empty data.
    """
    encoder_config = {"depth": 3, "width": 128}
    xattn_config = {"frequency": 2}
    num_buckets = 4

    pretrain_metadata = {
        "user_config": {
            "encoder": encoder_config,
            "cross_attention": xattn_config,
            "num_buckets": num_buckets,
        },
    }
    sft_metadata = {
        "encoder_config": encoder_config,
        "xattn_config": xattn_config,
        "num_buckets": num_buckets,
        "user_config": {},
    }

    pretrain_result = ccp._checkpoint_load_bearing_metadata(pretrain_metadata)
    sft_result = ccp._checkpoint_load_bearing_metadata(sft_metadata)

    assert pretrain_result == sft_result
    assert pretrain_result["encoder_config"] == encoder_config
    assert pretrain_result["xattn_config"] == xattn_config
    assert pretrain_result["num_buckets"] == num_buckets
    assert pretrain_result["user_config"] == {}


def test_checkpoint_load_bearing_metadata_recovers_nested_pretrain_encoder_config() -> None:
    """Regression for the original bug: a pretrain-layout metadata dict
    (encoder/cross_attention nested under ``user_config``, no top-level
    mirror) used to read ``encoder_config``/``xattn_config`` as ``None`` —
    silently dropping load-bearing training config from the fingerprint
    entirely, even though the checkpoint genuinely carries it.
    """
    metadata = {
        "user_config": {
            "encoder": {"depth": 3},
            "cross_attention": {"frequency": 2},
        },
    }
    result = ccp._checkpoint_load_bearing_metadata(metadata)
    assert result["encoder_config"] == {"depth": 3}
    assert result["xattn_config"] == {"frequency": 2}


def test_checkpoint_load_bearing_metadata_rejects_missing_user_config() -> None:
    with pytest.raises(ClusterFeatureContractError, match="requires a user_config mapping"):
        ccp._checkpoint_load_bearing_metadata({})
    with pytest.raises(ClusterFeatureContractError, match="requires a user_config mapping"):
        ccp._checkpoint_load_bearing_metadata({"user_config": "not-a-mapping"})


# --------------------------------------------------------------------------
# ``_load_json_mapping`` — raise sites, plus the payload-read swap-race
# regression that is unique to the fused (constraint 1) design: the gap
# between the fd-bound hash's close and the necessary second (payload) open
# is not covered by B1's own primitive-level tests, since that fd is never
# opened by ``stream_hash_file_stable`` itself.
# --------------------------------------------------------------------------


def test_load_json_mapping_returns_dict_and_digest_matching_a_direct_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    raw, digest = ccp._load_json_mapping(path, surface="test payload")
    assert raw == {"a": 1}
    expected = cci.stream_hash_file_stable(path)
    assert digest.sha256 == expected.sha256
    assert digest.byte_length == expected.byte_length


def test_load_json_mapping_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ClusterFeatureContractError, match="must contain a JSON object"):
        ccp._load_json_mapping(path, surface="test payload")


def test_load_json_mapping_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ClusterFeatureContractError, match="could not read test payload"):
        ccp._load_json_mapping(path, surface="test payload")


def test_load_json_mapping_wraps_payload_read_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "meta.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    resolved = path.resolve()
    real_read_bytes = Path.read_bytes

    def raising_read_bytes(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            raise OSError("simulated read failure")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", raising_read_bytes)
    with pytest.raises(ClusterFeatureContractError, match="could not read test payload"):
        ccp._load_json_mapping(path, surface="test payload")


def test_load_json_mapping_detects_a_real_equal_size_swap_around_the_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the deviation's own new gap-closure: a swap staged around the
    *second* open (the payload read that happens after the fused fd-bound
    hash has already completed and closed) is still caught, because the
    payload's own SHA-256 is verified against the digest from the first
    open — content-equality, not a second independent pathname-stat
    snapshot. Same swap-then-restore schedule as B1's own proof pattern
    (equal-size A->B before open, B->A' restore after), targeted at the
    second access to the path rather than the first.
    """
    path = tmp_path / "meta.json"
    a_bytes = json.dumps({"a": 1}).encode("utf-8")
    path.write_bytes(a_bytes)
    resolved = path.resolve()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^test payload changed while it was being loaded: {re.escape(str(resolved))}$",
    ):
        ccp._load_json_mapping(path, surface="test payload")
    assert open_count["n"] >= 2
    # Left on disk: the restored original content — the substitution was
    # transient and would be invisible to anything checking the path only
    # after the fact.
    assert resolved.read_bytes() == a_bytes


# --------------------------------------------------------------------------
# ``resolve_eval_checkpoint_directory`` — trim-by-import (verbatim port; the
# oracle test uses only B2a symbols and constructs its own tmp dirs).
# --------------------------------------------------------------------------


def test_checkpoint_directory_resolution_uses_public_runtime_utility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = tmp_path / "direct"
    direct.mkdir()
    assert resolve_eval_checkpoint_directory(
        checkpoint_dir=direct,
        checkpoint_source=None,
        checkpoint_model_tag=None,
    ) == (direct.resolve(), "direct")

    resolved = tmp_path / "resolved"
    resolved.mkdir()
    calls: list[tuple[str, str | None]] = []

    def fake_get_checkpoint_dir(source: str, model_tag: str | None = None) -> str:
        calls.append((source, model_tag))
        return str(resolved)

    monkeypatch.setattr(ccp, "get_checkpoint_dir", fake_get_checkpoint_dir)
    assert resolve_eval_checkpoint_directory(
        checkpoint_dir=None,
        checkpoint_source="encdec",
        checkpoint_model_tag="clustered",
    ) == (resolved.resolve(), "clustered")
    assert calls == [("encdec", "clustered")]
    with pytest.raises(ClusterFeatureContractError, match="exactly one"):
        resolve_eval_checkpoint_directory(
            checkpoint_dir=direct,
            checkpoint_source="encdec",
            checkpoint_model_tag="clustered",
        )


def test_checkpoint_directory_resolution_rejects_blank_source_and_tag(
    tmp_path: Path,
) -> None:
    with pytest.raises(ClusterFeatureContractError, match="non-empty string"):
        resolve_eval_checkpoint_directory(
            checkpoint_dir=None,
            checkpoint_source="",
            checkpoint_model_tag="clustered",
        )
    with pytest.raises(ClusterFeatureContractError, match="must be explicit"):
        resolve_eval_checkpoint_directory(
            checkpoint_dir=None,
            checkpoint_source="encdec",
            checkpoint_model_tag=None,
        )


def test_checkpoint_directory_resolution_wraps_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_get_checkpoint_dir(source: str, model_tag: str | None = None) -> str:
        raise ValueError("no such checkpoint")

    monkeypatch.setattr(ccp, "get_checkpoint_dir", raising_get_checkpoint_dir)
    with pytest.raises(ClusterFeatureContractError, match="could not resolve clustered checkpoint"):
        resolve_eval_checkpoint_directory(
            checkpoint_dir=None,
            checkpoint_source="encdec",
            checkpoint_model_tag="clustered",
        )


# --------------------------------------------------------------------------
# ``resolve_checkpoint_pin`` — trim-by-import (2 verbatim) + new raise-site
# coverage + checkpoint-level swap-race regressions (constraint 1).
# --------------------------------------------------------------------------


def test_checkpoint_pin_freezes_last_step_and_hashes_full_model(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=1)
    _write_checkpoint(checkpoint_dir, step=7)

    pin, _, _ = resolve_checkpoint_pin(checkpoint_dir, step=None)
    assert pin.step == 7
    assert pin.feature_vocab == FeatureVocab(8, (1, 2, 3, 4, 5, 6))
    expected = cci.stream_hash_file_stable(checkpoint_dir / "model_000007.pt")
    assert pin.model_file.sha256 == expected.sha256

    _write_checkpoint(checkpoint_dir, step=9)
    assert pin.step == 7


def test_checkpoint_pin_rejects_identity_vocab_buffers(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(
        checkpoint_dir,
        step=1,
        ids=tuple(range(8)),
    )
    with pytest.raises(ClusterFeatureContractError, match="identity"):
        resolve_checkpoint_pin(checkpoint_dir, step=1)


def test_resolve_checkpoint_pin_rejects_bool_dtype_vocab_ids_buffer(tmp_path: Path) -> None:
    """Checkpoint preflight rejects boolean IDs with its public contract error.

    The shared vocabulary parser also rejects bool. These values would form
    a plausible compact vocabulary if silently coerced to integer IDs.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((2, 2)),
            VOCAB_IDS_KEY: t.tensor([False, True], dtype=t.bool),
            VOCAB_N_SAE_KEY: t.tensor(8, dtype=t.int64),
        },
        checkpoint_dir / "model_000000.pt",
    )
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"user_config": {}}), encoding="utf-8"
    )
    with pytest.raises(
        ClusterFeatureContractError, match="must be an integer tensor, got torch.bool"
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=0)


def test_resolve_checkpoint_pin_rejects_non_directory(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(ClusterFeatureContractError, match="is not a directory"):
        resolve_checkpoint_pin(not_a_dir, step=0)


def test_resolve_checkpoint_pin_rejects_negative_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    with pytest.raises(ClusterFeatureContractError, match="non-negative strict int"):
        resolve_checkpoint_pin(checkpoint_dir, step=-1)


def test_resolve_checkpoint_pin_wraps_torch_load_failure(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    # Not a valid torch checkpoint (garbage bytes) -> t.load(..., mmap=True)
    # raises RuntimeError, which resolve_checkpoint_pin must wrap.
    (checkpoint_dir / "model_000000.pt").write_bytes(b"not a real torch checkpoint file")
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"user_config": {}}), encoding="utf-8"
    )
    with pytest.raises(
        ClusterFeatureContractError, match="could not safely load checkpoint state dict"
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=0)


def test_resolve_checkpoint_pin_rejects_non_mapping_state_dict(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    t.save(t.zeros(3), checkpoint_dir / "model_000000.pt")
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"user_config": {}}), encoding="utf-8"
    )
    with pytest.raises(ClusterFeatureContractError, match="did not contain a state-dict mapping"):
        resolve_checkpoint_pin(checkpoint_dir, step=0)


def test_resolve_checkpoint_pin_rejects_width_exceeding_max_safe_width(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    oversized = ccp.MAX_SAFE_SELECTED_FEATURE_WIDTH + 1
    t.save(
        {FEATURE_EMBED_KEY: t.zeros((oversized, 2))},
        checkpoint_dir / "model_000000.pt",
    )
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"user_config": {}}), encoding="utf-8"
    )
    with pytest.raises(ClusterFeatureContractError, match="exceeds 65536"):
        resolve_checkpoint_pin(checkpoint_dir, step=0)


def test_resolve_checkpoint_pin_strips_orig_mod_prefix_from_a_compiled_checkpoint(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 4 (MEDIUM): a checkpoint saved from a
    ``torch.compile``-wrapped model (without stripping first) carries EVERY
    state-dict key prefixed ``_orig_mod.`` (e.g. ``_orig_mod.encoder.
    feature_embed.weight``) — ``resolve_checkpoint_pin`` must still recover
    the SAME feature vocabulary an uncompiled checkpoint would, not reject
    it as missing an encoder entirely (the independent loader's own
    pre-fix failure mode: ``feature_vocab_from_state_dict`` would raise
    "state dict is missing 'encoder.feature_embed.weight'").
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    ids = (1, 2, 3, 4, 5, 6)
    n_sae_features = 8
    state = {
        FEATURE_EMBED_KEY: t.zeros((len(ids), 2)),
        VOCAB_IDS_KEY: t.tensor(ids, dtype=t.int64),
        VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
    }
    compiled_state = {f"_orig_mod.{k}": v for k, v in state.items()}
    assert not any(key.startswith("_orig_mod.") for key in state)  # sanity: prefix genuinely added
    t.save(compiled_state, checkpoint_dir / "model_000000.pt")
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"user_config": {}}), encoding="utf-8"
    )

    pin, _metadata, model_data = resolve_checkpoint_pin(checkpoint_dir, step=0)

    assert pin.n_sae_features == n_sae_features
    assert pin.feature_vocab == FeatureVocab(n_sae_features, ids)
    assert not any(key.startswith("_orig_mod.") for key in model_data)
    assert FEATURE_EMBED_KEY in model_data


def test_resolve_checkpoint_pin_detects_a_real_equal_size_metadata_swap_around_the_fused_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constraint 1's checkpoint-level swap-race regression, exact same
    proof pattern as B1's (equal-size A->B swap staged immediately before
    ``open()``, restored to fresh A-content immediately after), now
    exercised end-to-end through ``resolve_checkpoint_pin`` rather than
    directly against ``stream_hash_file_stable`` — proving the fix carries
    through the delegation (``resolve_checkpoint_pin`` ->
    ``_load_json_mapping`` -> ``stream_hash_file_stable`` ->
    ``stream_hash_file_descriptor_bound``) for the checkpoint metadata file.

    Per B1's own reviewed correctness note: this equal-size, fresh-inode
    restore schedule does not, on its own, prove a pathname-only
    before/after ``stat()`` pair would miss it (a fresh inode also changes
    ``st_ino``) — see
    ``test_immutable_io.py::test_stream_hash_file_descriptor_bound_detects_a_real_parent_directory_swap``
    for that structural proof at the primitive level. This test only proves
    delegation: the fix is carried all the way through
    ``resolve_checkpoint_pin``, not just present in the low-level helper.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=1)
    metadata_path = (checkpoint_dir / "meta_000001.json").resolve()
    a_bytes = metadata_path.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == metadata_path and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, metadata_path)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, metadata_path)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(metadata_path))}$",
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=1)
    assert swapped["done"]


def test_resolve_checkpoint_pin_detects_a_real_equal_size_metadata_swap_around_the_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to the fused-hash regression above: the swap this time
    targets the *second* open of the metadata path (the payload read inside
    ``_load_json_mapping``, which runs after the fd-bound hash already
    closed) — the gap unique to this deviation's two-open design, not
    covered by B1's own primitive-level tests (that fd is never opened by
    ``stream_hash_file_stable`` itself). Caught via content-hash
    verification against the digest from the first open, not a second
    independent pathname-stat snapshot.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=1)
    metadata_path = (checkpoint_dir / "meta_000001.json").resolve()
    a_bytes = metadata_path.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == metadata_path:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, metadata_path)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, metadata_path)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            "^checkpoint metadata changed while it was being loaded: "
            f"{re.escape(str(metadata_path))}$"
        ),
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=1)
    assert open_count["n"] >= 2


# --------------------------------------------------------------------------
# ``_hash_and_load_checkpoint_model`` — one file object for hash AND load
# (Codex fix batch finding 2, HIGH). The probe below empirically establishes
# WHY mmap=True cannot be fd-bound on this torch build (justifying the
# eager mmap=False fallback the module docstring documents); the remaining
# three tests prove the fd-bound design closes the symlink-retarget and
# equal-size-swap windows finding 2 reported, plus sonnet-8's widened
# metadata cold-check window.
# --------------------------------------------------------------------------


def test_probe_torch_load_mmap_rejects_a_file_object(tmp_path: Path) -> None:
    """Empirical probe pinning this torch build's actual behavior: ``t.load(
    <open file object>, ..., mmap=True)`` is rejected outright, not merely
    unbound to the descriptor. If a future torch upgrade makes this
    SUCCEED, ``_hash_and_load_checkpoint_model``'s eager (``mmap=False``)
    fallback -- and the "accepted residual: none" claim in the module
    docstring -- should be revisited: a raised ``ValueError`` here means
    "mmap=True never got the chance to leave the load unbound"; a change to
    this test's outcome is the tripwire for that revisit.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=0)
    model_path = checkpoint_dir / "model_000000.pt"

    with model_path.open("rb") as handle, pytest.raises(ValueError, match="mmap"):
        t.load(handle, map_location="cpu", weights_only=True, mmap=True)


def test_hash_and_load_checkpoint_model_is_immune_to_a_symlink_retarget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symlink-retarget: ``model_path`` is a symlink to file A. Retargeting
    it to file B immediately after the function's one ``resolve(strict=
    True)`` call (but before the open/hash/load) has NO EFFECT: the resolve
    happens exactly once, and every subsequent operation touches the
    already-resolved concrete inode, never re-consulting the symlink. This
    is PROVABLY IMPOSSIBLE, not merely detected after the fact -- B is
    never read at all, and the result is deterministically A's.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    a_target = tmp_path / "model_a.pt"
    b_target = tmp_path / "model_b.pt"
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((6, 2)),
            VOCAB_IDS_KEY: t.tensor((1, 2, 3, 4, 5, 6), dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(8, dtype=t.int64),
        },
        a_target,
    )
    t.save(
        {
            FEATURE_EMBED_KEY: t.ones((3, 2)),
            VOCAB_IDS_KEY: t.tensor((0, 1, 2), dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(3, dtype=t.int64),
        },
        b_target,
    )
    model_path = checkpoint_dir / "model_000000.pt"
    model_path.symlink_to(a_target)

    real_resolve = Path.resolve
    retargeted = {"done": False}

    def retargeting_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        result = real_resolve(self, *args, **kwargs)
        if self == model_path and not retargeted["done"]:
            retargeted["done"] = True
            model_path.unlink()
            model_path.symlink_to(b_target)
        return result

    monkeypatch.setattr(Path, "resolve", retargeting_resolve)

    digest, model_data = ccp._hash_and_load_checkpoint_model(model_path)

    assert retargeted["done"]
    assert t.equal(model_data[FEATURE_EMBED_KEY], t.zeros((6, 2)))  # A's content, not B's
    assert digest.sha256 == cci.stream_hash_file_stable(a_target).sha256


def test_resolve_checkpoint_pin_detects_a_real_equal_size_model_swap_around_the_hash_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A->B->A swap AROUND the (now fused) hash-and-load: same swap-then-
    restore schedule as the metadata regressions above, targeted at the
    model file's single ``open()``. Proves the fix is carried through
    ``resolve_checkpoint_pin`` end-to-end, and that the swap is caught
    whether it lands before the descriptor-bound fstat check or during the
    read (both collapse to the same message; see ``_hash_and_load_
    checkpoint_model``).
    """
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=1)
    model_path = (checkpoint_dir / "model_000001.pt").resolve()
    a_bytes = model_path.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_model_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_model_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == model_path and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, model_path)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, model_path)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            "^checkpoint model file changed while it was being hashed: "
            f"{re.escape(str(model_path))}$"
        ),
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=1)
    assert swapped["done"]


def test_resolve_checkpoint_pin_detects_a_metadata_mutation_staged_during_the_model_hash_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sonnet-8: the metadata cold-check (``_assert_digest_stat_unchanged
    (metadata_digest)``) runs *after* ``_hash_and_load_checkpoint_model``,
    not immediately after ``_load_json_mapping`` returns -- a DELIBERATE
    WIDENING of the metadata-mutation-detection window (see the module
    docstring), not the oracle's narrower position. This locks the widened
    window in with a real, PERMANENT (not swap-and-restore) mutation staged
    exactly in the gap a narrower ordering would miss: after
    ``_load_json_mapping`` has already returned a fresh digest, but before
    either cold check runs. A future refactor that moves the metadata cold
    check back to immediately after ``_load_json_mapping`` (the oracle's
    position) would stop observing this mutation and fail this test.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint_dir, step=1)
    metadata_path = (checkpoint_dir / "meta_000001.json").resolve()
    real_hash_and_load = ccp._hash_and_load_checkpoint_model

    def mutating_hash_and_load(path: Path) -> Any:
        result = real_hash_and_load(path)
        original = metadata_path.read_bytes()
        metadata_path.write_bytes(b"X" * len(original))  # permanent, equal-size mutation
        return result

    monkeypatch.setattr(ccp, "_hash_and_load_checkpoint_model", mutating_hash_and_load)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(f"^pinned contract input changed after hashing: {re.escape(str(metadata_path))}$"),
    ):
        resolve_checkpoint_pin(checkpoint_dir, step=1)


# --------------------------------------------------------------------------
# ``CheckpointPin`` — canonical-JSON strict-domain round-trips (constraint 4).
# --------------------------------------------------------------------------


def test_checkpoint_pin_semantic_and_provenance_dicts_pass_canonical_json_strict_domain(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    pin = _pin(tmp_path / "checkpoint", mask)

    semantic_json = cci._canonical_json(pin.semantic_dict())
    assert json.loads(semantic_json) == pin.semantic_dict()

    provenance_json = cci._canonical_json(pin.provenance_dict())
    assert json.loads(provenance_json) == pin.provenance_dict()

    # An identity-normalized (None) vocab must also serialize cleanly.
    pin_no_vocab = replace(pin, feature_vocab=None)
    no_vocab_json = cci._canonical_json(pin_no_vocab.semantic_dict())
    assert json.loads(no_vocab_json)["feature_vocab"] == {
        "persistent": False,
        "n_sae_features": 8,
    }


def test_checkpoint_pin_load_bearing_metadata_with_non_finite_float_raises_via_canonical_json(
    tmp_path: Path,
) -> None:
    """Reported finding (constraint 4): a checkpoint metadata file carrying
    a non-finite float (NaN/Infinity — a JSON extension Python's own
    ``json.loads`` accepts by default) anywhere inside a load-bearing region
    now raises ``ClusterFeatureContractError`` from B1's strict
    ``_canonical_json``, instead of the oracle's silent NaN-token
    pass-through (the oracle's own ``_canonical_json`` used bare
    ``json.dumps`` with no ``allow_nan=False`` and no pre-validation, so it
    would have re-emitted the same non-standard ``NaN`` token rather than
    failing loudly). This is intentional hardening inherited from B1, not a
    silent workaround: a checkpoint whose recorded training config contains
    a non-finite value is itself anomalous and should fail loudly rather
    than propagate a token that is not valid JSON.
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((6, 2)),
            VOCAB_IDS_KEY: t.tensor((1, 2, 3, 4, 5, 6), dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(8, dtype=t.int64),
        },
        checkpoint_dir / "model_000000.pt",
    )
    (checkpoint_dir / "meta_000000.json").write_text(
        json.dumps({"model_config": {"foo": float("nan")}, "user_config": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ClusterFeatureContractError, match="non-finite float"):
        resolve_checkpoint_pin(checkpoint_dir, step=0)


# --------------------------------------------------------------------------
# ``_assert_checkpoint_agreement`` — the agreement matrix.
#
# Edges checked by the function, and the fault-class matrix cell each case
# below exercises (rows: MISSING / EXTRA / MISMATCHED / MALFORMED; columns:
# checkpoint metadata (JSON) <-> config, persistent vocabulary (state-dict)
# <-> config, checkpoint metadata's vocab MIRROR <-> persistent vocabulary,
# checkpoint metadata's cluster descriptor/canonical <-> config, checkpoint
# width <-> config):
#
#   Edge A  metadata.feature_mask_canonical <-> config.feature_mask
#     MISSING     -> missing_mask, empty_metadata_mapping
#     MISMATCHED  -> different_mask
#   Edge B  checkpoint.feature_vocab (persistent) <-> config-expected vocab
#     (config-expected is the current-effective vocabulary (1, 3, 4, 6) —
#     see _base_agreement_fixture's docstring)
#     MISSING     -> state_vocab_subset (proper subset (1, 3, 4): missing
#                    member 6 only, no extras), checkpoint_vocab_entirely_absent (total)
#     EXTRA       -> state_vocab_superset
#     MISMATCHED  -> state_vocab_mixed_missing_and_extra (neither a pure
#                    subset nor a pure superset: missing member 6 while also
#                    carrying extras 2 and 5 the config does not expect —
#                    Codex pass-2 residual finding 2, its own labeled cell
#                    distinct from the missing-only case above)
#   Edge C  metadata.feature_vocab (mirror) <-> checkpoint.feature_vocab (persistent)
#     MISSING     -> test_checkpoint_agreement_tolerates_missing_metadata_vocab_mirror (NOT a raise — the mirror is optional)
#     MISMATCHED  -> metadata_vocab_mismatch
#   Edge D  metadata.feature_clusters_descriptor/canonical <-> config.feature_clusters
#     MISSING     -> missing_cluster_descriptor, missing_cluster_canonical
#     MISMATCHED  -> same_vocab_different_cluster
#     MALFORMED   -> malformed_cluster_descriptor
#   Edge E  checkpoint.n_sae_features <-> config.feature_clusters.n_sae_features
#     MISMATCHED  -> original_width_mismatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_mask", "feature_mask_canonical"),
        ("empty_metadata_mapping", "feature_mask_canonical"),
        ("different_mask", "feature_mask_canonical"),
        ("state_vocab_subset", "state-dict FeatureVocab"),
        ("state_vocab_mixed_missing_and_extra", "state-dict FeatureVocab"),
        ("state_vocab_superset", "state-dict FeatureVocab"),
        ("checkpoint_vocab_entirely_absent", "state-dict FeatureVocab"),
        ("metadata_vocab_mismatch", "metadata mirror"),
        ("missing_cluster_descriptor", "lacks complete"),
        ("missing_cluster_canonical", "lacks complete"),
        ("malformed_cluster_descriptor", "invalid static feature-cluster"),
        ("same_vocab_different_cluster", "feature cluster mismatch"),
        ("original_width_mismatch", "original SAE width"),
    ],
)
def test_checkpoint_agreement_rejects_every_one_sided_or_malformed_identity(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    feature_mask, checkpoint, feature_clusters = _base_agreement_fixture(tmp_path)
    metadata = _agreement_metadata(feature_mask, checkpoint, feature_clusters)
    if case == "missing_mask":
        del metadata["feature_mask_canonical"]
    elif case == "empty_metadata_mapping":
        metadata = {}
    elif case == "different_mask":
        metadata["feature_mask_canonical"] = {
            "mode": "whitelist",
            "feature_ids": [1, 2, 3],
        }
    elif case == "state_vocab_subset":
        # Codex pass-2 residual finding 2 (LOW): a TRUE missing-only case —
        # (1, 3, 4) is a proper subset of the config-expected current-
        # effective vocabulary (1, 3, 4, 6) (missing member 6, no extras).
        # This value used to be (1, 2, 3, 4, 5), which was a genuine subset
        # of the PRE-#759 mask-only vocab (1, 2, 3, 4, 5, 6) the fixture
        # used before Codex fix batch finding 1's effective-vocabulary
        # rewrite; against the current effective vocab (1, 3, 4, 6) that
        # same value is actually MIXED (missing 6, extra 2 and 5) — see
        # state_vocab_mixed_missing_and_extra below, which now covers that
        # shape explicitly as its own cell instead of silently masquerading
        # as this one.
        checkpoint = replace(
            checkpoint,
            feature_vocab=FeatureVocab(8, (1, 3, 4)),
        )
    elif case == "state_vocab_mixed_missing_and_extra":
        # Codex pass-2 residual finding 2 (LOW): neither a pure subset nor
        # a pure superset of the config-expected (1, 3, 4, 6) — missing
        # member 6 while also carrying extras 2 and 5 the config does not
        # expect. This is the value state_vocab_subset held before this
        # fix; kept as its own labeled MISMATCHED cell rather than folded
        # back into the missing-only case above.
        checkpoint = replace(
            checkpoint,
            feature_vocab=FeatureVocab(8, (1, 2, 3, 4, 5)),
        )
    elif case == "state_vocab_superset":
        checkpoint = replace(
            checkpoint,
            feature_vocab=FeatureVocab(8, (0, 1, 2, 3, 4, 5, 6)),
        )
    elif case == "checkpoint_vocab_entirely_absent":
        checkpoint = replace(checkpoint, feature_vocab=None)
    elif case == "metadata_vocab_mismatch":
        metadata["feature_vocab"] = FeatureVocab(8, (1, 2, 3)).metadata_dict()
    elif case == "missing_cluster_descriptor":
        del metadata["feature_clusters_descriptor"]
    elif case == "missing_cluster_canonical":
        del metadata["feature_clusters_canonical"]
    elif case == "malformed_cluster_descriptor":
        metadata["feature_clusters_descriptor"] = {"schema_version": 1}
    elif case == "same_vocab_different_cluster":
        metadata["feature_clusters_canonical"] = {
            "schema_version": 1,
            "clusters": [[1, 2, 7], [4, 5]],
        }
    elif case == "original_width_mismatch":
        checkpoint = replace(
            checkpoint,
            n_sae_features=7,
            feature_vocab=FeatureVocab(7, (1, 2, 3, 4, 5, 6)),
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(ClusterFeatureContractError, match=match):
        ccp._assert_checkpoint_agreement(
            checkpoint=checkpoint,
            metadata=metadata,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
        )


def test_checkpoint_agreement_tolerates_missing_metadata_vocab_mirror(tmp_path: Path) -> None:
    """Edge C, MISSING: the checkpoint metadata's ``feature_vocab`` mirror
    is optional best-effort provenance, not a required field — a checkpoint
    that never wrote it (e.g. an older writer) must NOT fail agreement as
    long as everything else (mask, persistent vocab, clusters) agrees.
    """
    feature_mask, checkpoint, feature_clusters = _base_agreement_fixture(tmp_path)
    metadata = _agreement_metadata(feature_mask, checkpoint, feature_clusters)
    del metadata["feature_vocab"]

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
    )


# --------------------------------------------------------------------------
# ``_assert_checkpoint_agreement`` — explicit-null vocab mirror, key-
# presence semantics (Codex fix batch finding 5, MEDIUM). Both directions:
# a PRESENT ``null`` disagreeing with a compact vocab must reject (unlike
# an ABSENT key, tolerated above); a PRESENT ``null`` agreeing with an
# uncompacted (``None``) vocab must still pass.
# --------------------------------------------------------------------------


def test_checkpoint_agreement_rejects_explicit_null_vocab_mirror_against_a_compact_vocabulary(
    tmp_path: Path,
) -> None:
    """Direction 1 (reject): an explicit ``"feature_vocab": null`` in the
    checkpoint metadata is a WRITER asserting "no vocab" — which directly
    contradicts a compact (non-``None``) persistent state-dict
    vocabulary. ``metadata.get(...)`` cannot distinguish this from an
    absent key (both return ``None``), which is exactly how the original
    bug let a contradictory present mirror pass silently.
    """
    feature_mask, checkpoint, feature_clusters = _base_agreement_fixture(tmp_path)
    metadata = _agreement_metadata(feature_mask, checkpoint, feature_clusters)
    assert checkpoint.feature_vocab is not None  # base fixture is genuinely compacted
    metadata["feature_vocab"] = None  # present key, explicit null

    with pytest.raises(ClusterFeatureContractError, match="metadata mirror"):
        ccp._assert_checkpoint_agreement(
            checkpoint=checkpoint,
            metadata=metadata,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
        )


def test_checkpoint_agreement_accepts_explicit_null_vocab_mirror_against_no_vocab(
    tmp_path: Path,
) -> None:
    """Direction 2 (accept): an explicit ``"feature_vocab": null`` mirror
    AGREEING with an uncompacted (``None``) persistent vocabulary must
    still PASS — the fix's key-presence check compares a present value,
    it does not blanket-reject every ``None`` mirror regardless of
    whether it agrees.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(8)))  # identity mask -> None vocab
    checkpoint = _pin(tmp_path / "checkpoint", mask)
    assert checkpoint.feature_vocab is None
    metadata = _agreement_metadata(mask, checkpoint, None)
    metadata["feature_vocab"] = None  # present key, explicit null, AGREEING

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=mask,
        feature_clusters=None,
    )


# --------------------------------------------------------------------------
# Effective-vocabulary agreement: accept the mask+cluster vocabulary, reject
# mask-only vocabulary when clusters narrow it, accept cluster-only narrowing,
# and preserve inert-cluster and no-cluster behavior.
# --------------------------------------------------------------------------


def test_checkpoint_agreement_accepts_current_effective_vocabulary_with_clusters(
    tmp_path: Path,
) -> None:
    """current-effective/pass: mask ``(1..6)`` + clusters ``((1,2),(4,5))``
    narrow the persistent vocabulary to ``(1,3,4,6)`` — both mask AND
    cluster consolidation contribute. A checkpoint correctly carrying that
    effective vocabulary must PASS agreement.
    """
    feature_mask, checkpoint, feature_clusters = _base_agreement_fixture(tmp_path)
    metadata = _agreement_metadata(feature_mask, checkpoint, feature_clusters)

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
    )


def test_checkpoint_agreement_rejects_pre_effective_mask_only_vocabulary(
    tmp_path: Path,
) -> None:
    """pre-effective/reject: a checkpoint whose persistent vocabulary is the
    mask-only shape ``(1..6)`` despite clusters being configured to narrow
    it further predates effective-vocabulary training (#759). Mask,
    metadata mirror, and persistent vocab are all mutually self-consistent
    (exactly what an old, cluster-blind writer would have produced) — only
    the NEW cluster-aware comparison catches the regime mismatch, with an
    actionable message naming it rather than a generic disagreement.
    """
    feature_mask, checkpoint, feature_clusters = _base_agreement_fixture(tmp_path)
    mask_only_vocab = FeatureVocab.from_mask(feature_mask, checkpoint.n_sae_features)
    assert mask_only_vocab != checkpoint.feature_vocab  # genuinely narrower when correct
    checkpoint = replace(checkpoint, feature_vocab=mask_only_vocab)
    metadata = _agreement_metadata(feature_mask, checkpoint, feature_clusters)

    with pytest.raises(ClusterFeatureContractError, match="predates effective-vocabulary training"):
        ccp._assert_checkpoint_agreement(
            checkpoint=checkpoint,
            metadata=metadata,
            feature_mask=feature_mask,
            feature_clusters=feature_clusters,
        )


def test_checkpoint_agreement_accepts_cluster_only_narrowing_with_no_mask(
    tmp_path: Path,
) -> None:
    """cluster-only-narrowing: ``feature_mask=None`` (every original id is
    allowed), so static-cluster consolidation is the ONLY source of
    narrowing. Exercises ``FeatureClusterConsolidation.build(..., feature_
    mask=None)`` distinctly from the mask+cluster combination cell above,
    and requires ``_assert_checkpoint_agreement``'s ``feature_mask``
    parameter to genuinely accept ``None`` (not just ``feature_clusters``).
    """
    clusters = _clusters(tmp_path / "clusters.yaml")
    consolidation = FeatureClusterConsolidation.build(clusters, None)
    expected_vocab = effective_feature_vocab(consolidation, 8)
    assert expected_vocab is not None and expected_vocab.size < 8

    checkpoint = _pin(tmp_path / "checkpoint", None, feature_clusters=clusters)
    assert checkpoint.feature_vocab == expected_vocab
    metadata = _agreement_metadata(None, checkpoint, clusters)

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=None,
        feature_clusters=clusters,
    )


def test_checkpoint_agreement_tolerates_inert_cluster_consolidation(tmp_path: Path) -> None:
    """inert-cluster: clusters are configured, but the mask excludes one
    member of every configured pair ``((1,2),(4,5))``, so each cluster has
    at most one mask-allowed survivor and consolidation moves nothing —
    the effective vocabulary is IDENTICAL to the mask-only vocabulary.
    Proves the fix does not spuriously reject when a configured
    consolidation happens to be a no-op.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 4, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")  # default ((1, 2), (4, 5))
    consolidation = FeatureClusterConsolidation.build(clusters, mask)
    assert consolidation.is_identity_after_mask

    mask_only_vocab = FeatureVocab.from_mask(mask, 8)
    checkpoint = _pin(tmp_path / "checkpoint", mask, feature_clusters=clusters)
    assert checkpoint.feature_vocab == mask_only_vocab
    metadata = _agreement_metadata(mask, checkpoint, clusters)

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=mask,
        feature_clusters=clusters,
    )


def test_checkpoint_agreement_no_cluster_legacy_unchanged(tmp_path: Path) -> None:
    """no-cluster legacy: ``feature_clusters=None`` entirely (E3 no-cluster
    identity) — agreement reduces to exactly the pre-fix mask-only
    comparison, unaffected by the effective-vocabulary derivation.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    checkpoint = _pin(tmp_path / "checkpoint", mask)  # feature_clusters=None default
    assert checkpoint.feature_vocab == FeatureVocab.from_mask(mask, 8)
    metadata = _agreement_metadata(mask, checkpoint, None)

    ccp._assert_checkpoint_agreement(
        checkpoint=checkpoint,
        metadata=metadata,
        feature_mask=mask,
        feature_clusters=None,
    )


# --------------------------------------------------------------------------
# ``_statistics_identity`` — mismatch / raise-site coverage plus a tamper
# (swap-race) regression, exercised through this function specifically.
# --------------------------------------------------------------------------


def test_statistics_identity_happy_path(tmp_path: Path) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    identity, directory_identity, stats = ccp._statistics_identity(
        feature_dir=feature_dir,
        statistics_path=statistics_path,
        n_sae_features=16_384,
    )
    assert identity.n_sae_features == 16_384
    assert identity.feature_id_domain == (0, 1, 2)
    assert directory_identity.layer == 8
    assert directory_identity.segment_size == 128
    assert directory_identity.ignore_bos_features is False
    assert stats.metadata.n_sae_features == 16_384


def test_statistics_identity_rejects_turn_delimited_feature_directory(
    tmp_path: Path,
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(
        tmp_path,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="clustered evaluation supports fixed_tokens feature directories only",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_path_outside_feature_dir(tmp_path: Path) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    with pytest.raises(ClusterFeatureContractError, match="must be the statistics artifact"):
        ccp._statistics_identity(
            feature_dir=other_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_stats_width_not_matching_checkpoint(tmp_path: Path) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    with pytest.raises(ClusterFeatureContractError, match="does not match checkpoint"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=8_192,
        )


def test_statistics_identity_rejects_path_width_mismatch(tmp_path: Path) -> None:
    # Path says "16k" (16384) but the stats file + checkpoint both agree on
    # 32768 — isolates the path/stats cross-check from the earlier
    # stats-vs-checkpoint check, which would otherwise fire first.
    feature_dir, statistics_path = _write_stats_feature_dir(
        tmp_path,
        n_sae_features=32_768,
        width=SAEWidth.W_16K,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="path width=16384, checkpoint width=32768",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=32_768,
        )


def test_statistics_identity_rejects_segment_size_mismatch(tmp_path: Path) -> None:
    # write_steering_stats hardcodes segment_size=128; mismatch the path.
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path, segment_size=64)
    with pytest.raises(ClusterFeatureContractError, match="stats segment_size=128, path=64"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_layer_mismatch(tmp_path: Path) -> None:
    # write_steering_stats hardcodes layer=8; mismatch the path.
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path, layer=5)
    with pytest.raises(ClusterFeatureContractError, match="stats layer=8, path=5"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_pooling_mismatch(tmp_path: Path) -> None:
    # write_steering_stats hardcodes pooling_mode="max_pool"; mismatch path.
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path, pooling=PoolingMode.mean_pool)
    with pytest.raises(
        ClusterFeatureContractError,
        match="stats pooling='max_pool', path='mean_pool'",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_segment_mode_metadata_path_mismatch(
    tmp_path: Path,
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    tensors = load_file(str(statistics_path))
    metadata = stats_io_module._read_metadata(statistics_path)
    metadata["segment_mode"] = SegmentMode.turn_delimited_fixed_tokens.value
    save_file(tensors, str(statistics_path), metadata=metadata)

    with pytest.raises(
        ClusterFeatureContractError,
        match=("stats segment_mode='turn_delimited_fixed_tokens', path='fixed_tokens'"),
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_ignore_bos_features_mismatch(tmp_path: Path) -> None:
    # write_steering_stats never sets ignore_bos_features -> dataclass
    # default False; mismatch the path (True).
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path, ignore_bos_features=True)
    with pytest.raises(
        ClusterFeatureContractError,
        match="ignore_bos_features does not match the feature-directory leaf",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_reports_every_mismatch_together(tmp_path: Path) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(
        tmp_path,
        segment_size=64,
        layer=5,
        pooling=PoolingMode.mean_pool,
        ignore_bos_features=True,
    )
    with pytest.raises(ClusterFeatureContractError) as exc_info:
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )
    message = str(exc_info.value)
    assert "stats segment_size=128, path=64" in message
    assert "stats layer=8, path=5" in message
    assert "stats pooling='max_pool', path='mean_pool'" in message
    assert "ignore_bos_features does not match" in message


def test_statistics_identity_rejects_missing_percentile_checksum(tmp_path: Path) -> None:
    feature_dir = build_feature_dir(
        tmp_path / "features",
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
        n_sae_features=16_384,
        max_mags={0: 1.0},
        absent=range(1, 16_384),
    )
    # No save_percentile_checksum call: stored_checksum stays None.
    with pytest.raises(
        ClusterFeatureContractError,
        match="requires feature statistics with a stored",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_stale_percentile_checksum(tmp_path: Path) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    save_percentile_checksum(statistics_path, "0" * 64)
    with pytest.raises(
        ClusterFeatureContractError,
        match="does not match the recomputed table",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def _bypass_percentile_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize ``build_percentile_table``/the stored-checksum comparison.

    Finding: ``_statistics_identity``'s own domain-validation loop ("is not
    an integer" / "not canonical decimal form" / "contains duplicates")
    runs *after* ``build_percentile_table`` and the percentile-checksum
    comparison in the same function. ``build_percentile_table`` already
    does ``int(fid_str)`` for every key (raising the same class of
    ``ValueError`` first for a genuinely non-integer key), and a
    non-canonical-but-parseable key like ``"01"`` changes the rebuilt
    table's checksum, so the stored-checksum comparison rejects it first.
    All three domain-loop checks are therefore effectively unreachable via
    any real end-to-end call — they are exercised here, as a deliberate
    regression-safety net for a future call-order refactor, only by also
    bypassing both earlier gates. Reported in the task report rather than
    silently dropped.

    (Codex fix batch finding 3: since ``_load_statistics_snapshot`` fused
    the stored-checksum read into the same call that parses ``stats``,
    each caller of this helper must ALSO make its own ``_load_statistics_
    snapshot`` mock return ``checksum="irrelevant"`` — there is no longer a
    separate ``load_percentile_checksum`` reference to intercept here.)
    """
    monkeypatch.setattr(ccp, "build_percentile_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(ccp, "compute_percentile_checksum", lambda *args, **kwargs: "irrelevant")


def test_statistics_identity_rejects_non_integer_feature_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    real_load_snapshot = ccp._load_statistics_snapshot

    def bad_load_snapshot(path: Path) -> Any:
        stats, _checksum, raw_feature_ids, digest = real_load_snapshot(path)
        bad = dict(stats.per_feature_stats)
        bad["not-an-int"] = next(iter(stats.per_feature_stats.values()))
        return replace(stats, per_feature_stats=bad), "irrelevant", raw_feature_ids, digest

    monkeypatch.setattr(ccp, "_load_statistics_snapshot", bad_load_snapshot)
    _bypass_percentile_gates(monkeypatch)
    with pytest.raises(ClusterFeatureContractError, match="is not an integer"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_non_canonical_feature_id_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    real_load_snapshot = ccp._load_statistics_snapshot

    def bad_load_snapshot(path: Path) -> Any:
        stats, _checksum, raw_feature_ids, digest = real_load_snapshot(path)
        bad = dict(stats.per_feature_stats)
        bad["01"] = next(iter(stats.per_feature_stats.values()))
        return replace(stats, per_feature_stats=bad), "irrelevant", raw_feature_ids, digest

    monkeypatch.setattr(ccp, "_load_statistics_snapshot", bad_load_snapshot)
    _bypass_percentile_gates(monkeypatch)
    with pytest.raises(ClusterFeatureContractError, match="not canonical decimal form"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_rejects_duplicate_feature_id_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The oracle's own dedup guard (``len(domain) != len(set(domain))``) is
    unreachable via any real ``dict[str, PerFeatureStat]`` input: the prior
    canonical-decimal-form check already makes surviving keys a bijection
    with their parsed ints, and dict keys are inherently unique strings. It
    is exercised here only via a hand-built non-dict ``Mapping`` whose
    iteration yields a duplicate key — a defensive check preserved verbatim
    from the oracle, not a reachable production path. Documented as a
    finding rather than silently dropped (constraint 5's empty-return-audit
    spirit, applied to a defensive-dead branch instead).
    """

    class _DuplicateKeyMapping:
        def __init__(self, stat: Any) -> None:
            self._stat = stat

        def __iter__(self):
            yield "5"
            yield "5"

        def __getitem__(self, key: str) -> Any:
            return self._stat

    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    real_load_snapshot = ccp._load_statistics_snapshot

    def bad_load_snapshot(path: Path) -> Any:
        stats, _checksum, raw_feature_ids, digest = real_load_snapshot(path)
        sample = next(iter(stats.per_feature_stats.values()))
        return (
            replace(stats, per_feature_stats=_DuplicateKeyMapping(sample)),
            "irrelevant",
            raw_feature_ids,
            digest,
        )

    monkeypatch.setattr(ccp, "_load_statistics_snapshot", bad_load_snapshot)
    _bypass_percentile_gates(monkeypatch)
    with pytest.raises(ClusterFeatureContractError, match="contains duplicates"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


# --------------------------------------------------------------------------
# ``_assert_raw_feature_ids_valid`` — Codex fix batch finding 6 (MEDIUM).
# --------------------------------------------------------------------------


def test_statistics_identity_rejects_a_real_duplicate_raw_feature_id_with_a_self_consistent_checksum(
    tmp_path: Path,
) -> None:
    """A REAL safetensors file (not the monkeypatched fake mapping the
    dead-branch test above uses) whose raw ``feature_ids`` tensor genuinely
    repeats an ID. Row 0 (feature 5) is duplicated into every per-feature
    tensor, so the file has 3 rows on disk (ids 5, 5, 7) but the stored
    ``percentile_checksum`` is the one honestly computed from the CORRECT
    2-feature ``{5, 7}`` table -- reconstruction collapses the duplicated
    rows back to that identical 2-entry dict (the extra row is byte-for-byte
    identical to the original it duplicates), so the checksum comparison
    alone cannot catch this tamper. Only the raw pre-collapse tensor check
    (which runs on the tensor BEFORE that collapse) does.
    """
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path, feature_ids=(5, 7))
    tensors = load_file(str(statistics_path))
    metadata = stats_io_module._read_metadata(statistics_path)
    assert tensors["feature_ids"].tolist() == [5, 7]

    per_feature_tensor_keys = (
        "feature_ids",
        "counts",
        "pct_segments_active",
        "mean_mag",
        "mean_mag_incl_zeros",
        "std_mag",
        "min_mag",
        "max_mag",
        "median_mag",
        "percentile_values",
    )
    tampered = dict(tensors)
    for key in per_feature_tensor_keys:
        original = tensors[key]
        tampered[key] = t.cat([original[:1], original[:1], original[1:]], dim=0)
    assert tampered["feature_ids"].tolist() == [5, 5, 7]
    save_file(tampered, str(statistics_path), metadata=metadata)

    with pytest.raises(
        ClusterFeatureContractError,
        match="statistics raw feature_ids tensor contains a duplicate feature ID",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_wraps_load_statistics_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)

    def raising_load_snapshot(path: Path) -> Any:
        raise ValueError("corrupt statistics file")

    monkeypatch.setattr(ccp, "_load_statistics_snapshot", raising_load_snapshot)
    with pytest.raises(ClusterFeatureContractError, match="could not load feature statistics"):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_wraps_percentile_table_rebuild_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)

    def raising_build_percentile_table(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("missing percentile key")

    monkeypatch.setattr(ccp, "build_percentile_table", raising_build_percentile_table)
    with pytest.raises(
        ClusterFeatureContractError,
        match="could not rebuild percentile table",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )


def test_statistics_identity_detects_a_real_equal_size_swap_around_the_fused_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Statistics-identity tamper test: same swap-then-restore schedule as
    B1's own proof pattern (and the checkpoint-level regressions above),
    targeted at the statistics file's initial ``stream_hash_file_stable``
    open inside ``_load_statistics_snapshot`` — proving the delegation
    carries through ``_statistics_identity`` too.
    """
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    resolved = statistics_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = bytes(len(a_bytes))  # zero-filled, equal size, different content
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, resolved)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )
    assert swapped["done"]


def test_statistics_identity_detects_a_real_equal_size_swap_around_the_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex fix batch finding 3 (HIGH): companion to the fused-hash
    regression above, targeting the SECOND open of the statistics path
    (the payload read inside ``_load_statistics_snapshot``, which runs
    after the fd-bound hash already closed) — this is the "former reopen
    point" that used to be ``load_statistics(path)``'s own independent
    safetensors + header open, one of the two reopens finding 3 reported.
    Caught via content-hash verification against the digest from the first
    open, not a second independent pathname-stat snapshot.
    """
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    resolved = statistics_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = bytes(len(a_bytes))  # zero-filled, equal size, different content
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            f"^feature statistics changed while it was being loaded: {re.escape(str(resolved))}$"
        ),
    ):
        ccp._statistics_identity(
            feature_dir=feature_dir,
            statistics_path=statistics_path,
            n_sae_features=16_384,
        )
    assert open_count["n"] >= 2


def test_load_statistics_snapshot_detects_a_real_equal_size_swap_around_the_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same proof directly at the ``_load_statistics_snapshot`` unit level
    (mirrors ``test_load_json_mapping_detects_a_real_equal_size_swap_
    around_the_payload_read``'s unit-level companion to the checkpoint
    metadata's own end-to-end regression)."""
    feature_dir, statistics_path = _write_stats_feature_dir(tmp_path)
    resolved = statistics_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = bytes(len(a_bytes))
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            f"^feature statistics changed while it was being loaded: {re.escape(str(resolved))}$"
        ),
    ):
        ccp._load_statistics_snapshot(statistics_path)
    assert open_count["n"] >= 2


# ============================================================================
# B2b: SAE resolve/validate, selection identity/manifest, selection contract
# ============================================================================


# --------------------------------------------------------------------------
# ``_default_sae_loader``
# --------------------------------------------------------------------------


def test_default_sae_loader_passes_resolved_identifiers_and_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = SimpleNamespace(
        release="gemma-scope-2b-res-canonical",
        sae_id="layer_8/width_16k/average_l0_71",
        hook_name="blocks.8.hook_resid_post",
    )
    calls: list[dict[str, Any]] = []
    fake_sae = SimpleNamespace(name="fake")

    def fake_from_pretrained(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return fake_sae

    monkeypatch.setattr(ccp.SAE, "from_pretrained", fake_from_pretrained)
    assert ccp._default_sae_loader(resolved) is fake_sae  # ty: ignore[invalid-argument-type]
    assert calls == [
        {
            "release": "gemma-scope-2b-res-canonical",
            "sae_id": "layer_8/width_16k/average_l0_71",
            "device": "cpu",
        }
    ]


def test_default_sae_loader_unwraps_a_tuple_return(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = SimpleNamespace(release="r", sae_id="s", hook_name="h")
    fake_sae = SimpleNamespace(name="fake")
    monkeypatch.setattr(ccp.SAE, "from_pretrained", lambda **_kwargs: (fake_sae, {"cfg_dict": {}}))
    assert ccp._default_sae_loader(resolved) is fake_sae  # ty: ignore[invalid-argument-type]


def test_default_sae_loader_returns_a_bare_sae_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = SimpleNamespace(release="r", sae_id="s", hook_name="h")
    fake_sae = SimpleNamespace(name="fake")
    monkeypatch.setattr(ccp.SAE, "from_pretrained", lambda **_kwargs: fake_sae)
    assert ccp._default_sae_loader(resolved) is fake_sae  # ty: ignore[invalid-argument-type]


# --------------------------------------------------------------------------
# ``_resolve_and_validate_sae`` — trim-by-import (oracle's parametrized shape
# test + decoder-row-width test) plus new raise-site coverage.
# --------------------------------------------------------------------------


def _sae_feature_directory() -> ccp.FeatureDirectoryIdentity:
    return ccp.FeatureDirectoryIdentity(
        segment_size=128,
        layer=8,
        activation_type="residual",
        pooling="max_pool",
        width="16k",
        model="gemma-2-2b",
        dataset_phase="base",
        context_independent_features=False,
        ignore_bos_features=True,
    )


def _sae_metadata(**sae_cfg_overrides: Any) -> dict[str, Any]:
    sae_cfg = {
        "release": "gemmascope",
        "model": "gemma-2-2b",
        "activation_type": "residual",
        "layer": 8,
        "width": "16k",
        "l0": 71,
    }
    sae_cfg.update(sae_cfg_overrides)
    return {"user_config": {"sae": sae_cfg}}


_SAE_RESOLVED = SimpleNamespace(
    release="gemma-scope-2b-res-canonical",
    sae_id="layer_8/width_16k/average_l0_71",
    hook_name="blocks.8.hook_resid_post",
)


@pytest.mark.parametrize(
    ("shape", "d_in", "hook_name", "match"),
    [
        ((2, 16384), 2, "blocks.8.hook_resid_post", "one decoder row"),
        ((16384, 3), 2, "blocks.8.hook_resid_post", "input dimension"),
        ((16384, 2), 2, None, "hook_name"),
    ],
)
def test_loaded_sae_validates_feature_axis_input_dimension_and_hook(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int],
    d_in: int,
    hook_name: str | None,
    match: str,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros(shape),
        cfg=SimpleNamespace(d_in=d_in, metadata=SimpleNamespace(hook_name=hook_name)),
    )
    with pytest.raises(ClusterFeatureContractError, match=match):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_loaded_sae_records_actual_decoder_row_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(d_in=2, metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post")),
    )
    identity = ccp._resolve_and_validate_sae(
        metadata=_sae_metadata(),
        feature_directory=_sae_feature_directory(),
        n_sae_features=16384,
        sae_loader=lambda _resolved: sae,
    )
    assert identity.actual_loaded_width == 16384
    assert identity.release == "gemma-scope-2b-res-canonical"
    assert identity.hook_name == "blocks.8.hook_resid_post"
    assert identity.width == "16k"


def test_resolve_and_validate_sae_rejects_missing_user_config() -> None:
    with pytest.raises(ClusterFeatureContractError, match="user_config is missing"):
        ccp._resolve_and_validate_sae(
            metadata={},
            feature_directory=_sae_feature_directory(),
            n_sae_features=8,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_non_mapping_sae_config() -> None:
    with pytest.raises(ClusterFeatureContractError, match="user_config.sae must be a mapping"):
        ccp._resolve_and_validate_sae(
            metadata={"user_config": {"sae": "not-a-mapping"}},
            feature_directory=_sae_feature_directory(),
            n_sae_features=8,
            sae_loader=lambda _resolved: None,
        )


# --------------------------------------------------------------------------
# ``_resolve_and_validate_sae`` — Codex combined-review finding 5 (MEDIUM):
# strict-domain coercion of sae.layer was incomplete.
# --------------------------------------------------------------------------


def test_resolve_and_validate_sae_rejects_a_boolean_layer() -> None:
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(layer=True),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_a_float_layer() -> None:
    """Codex's exact reproduction: plain ``int(...)`` silently truncated a
    checkpoint layer of 8.9 to 8, which then passed the feature-directory
    and hook agreement checks below."""
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(layer=8.9),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_a_numeric_string_layer() -> None:
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(layer="8"),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_a_negative_layer() -> None:
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(layer=-1),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_wraps_an_invalid_release_value() -> None:
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(release="not-a-real-release"),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_wraps_a_missing_config_key() -> None:
    metadata = _sae_metadata()
    del metadata["user_config"]["sae"]["width"]
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=metadata,
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_wraps_resolve_sae_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_resolve_sae(**_kwargs: Any) -> Any:
        raise ValueError("unsupported model/width/l0 combination")

    monkeypatch.setattr(ccp, "resolve_sae", raising_resolve_sae)
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint SAE config"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_feature_directory_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    mismatched_directory = replace(_sae_feature_directory(), model="gemma-3-4b")
    with pytest.raises(
        ClusterFeatureContractError, match="does not match the parsed feature-directory identity"
    ):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=mismatched_directory,
            n_sae_features=16384,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_rejects_configured_width_not_matching_checkpoint_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    with pytest.raises(
        ClusterFeatureContractError, match="configured width does not match original checkpoint"
    ):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=999,
            sae_loader=lambda _resolved: None,
        )


def test_resolve_and_validate_sae_wraps_sae_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)

    def raising_loader(_resolved: Any) -> Any:
        raise RuntimeError("network unavailable")

    with pytest.raises(ClusterFeatureContractError, match="could not load SAE"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=raising_loader,
        )


def test_resolve_and_validate_sae_rejects_a_non_tensor_w_dec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(W_dec=[1, 2, 3])
    with pytest.raises(ClusterFeatureContractError, match="rank-2 W_dec tensor"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_1d_w_dec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(W_dec=t.zeros(16384))
    with pytest.raises(ClusterFeatureContractError, match="rank-2 W_dec tensor"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_hook_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(d_in=2, metadata=SimpleNamespace(hook_name="blocks.9.hook_resid_post")),
    )
    with pytest.raises(ClusterFeatureContractError, match="does not match resolved hook"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


# --------------------------------------------------------------------------
# ``_resolve_and_validate_sae`` — Codex combined-review finding 2 (HIGH):
# loaded-SAE identity could disagree with the SAE the runtime actually uses.
# --------------------------------------------------------------------------


def test_resolve_and_validate_sae_rejects_a_conflicting_direct_hook_even_when_metadata_hook_is_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex's exact reproduction: the ORIGINAL port's combined hook lookup
    gave cfg.metadata.hook_name silent precedence over the direct
    cfg.hook_name, so a WRONG direct hook with a CORRECT metadata hook
    passed this preflight while runtime extraction — which reads the direct
    attribute first (axbench_base/extraction.py's _sae_hook_name) — would
    have used the wrong hook. Reading both separately and requiring
    agreement now rejects this.
    """
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            hook_name="blocks.99.hook_resid_post",  # WRONG direct hook
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),  # correct
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="disagrees with"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_wrong_loaded_direct_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex's reproduction, release side: a wrong loaded release/ID must
    now be rejected even though hook_name and width both still agree."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            release="gemma-scope-2b-res-71",  # WRONG vs _SAE_RESOLVED.release
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="does not match resolved release"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_wrong_loaded_neuronpedia_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex's reproduction, Neuronpedia-id side (SAELens 6.37+ layout, no
    direct cfg.release)."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    monkeypatch.setattr(
        ccp,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-2-2b", "8-gemmascope-res-16k"),
    )
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            metadata=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                neuronpedia_id="gemma-2-2b/8-gemmascope-res-71",  # WRONG
            ),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="does not match resolved Neuronpedia id"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_skips_neuronpedia_id_check_when_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Validate ... when available": an unpublished combo (resolver raises
    ValueError) must not itself fail the preflight -- it just means this
    specific cross-check cannot run."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)

    def _raising_resolve_neuronpedia_ids(**_kwargs: Any) -> Any:
        raise ValueError("not published")

    monkeypatch.setattr(ccp, "resolve_neuronpedia_ids", _raising_resolve_neuronpedia_ids)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            metadata=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                neuronpedia_id="anything",
            ),
        ),
    )
    identity = ccp._resolve_and_validate_sae(
        metadata=_sae_metadata(),
        feature_directory=_sae_feature_directory(),
        n_sae_features=16384,
        sae_loader=lambda _resolved: sae,
    )
    assert identity.hook_name == "blocks.8.hook_resid_post"


def test_resolve_and_validate_sae_rejects_a_wrong_loaded_d_sae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            d_sae=999,  # WRONG -- cfg lies about its own tensor's row count
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="d_sae/n_features does not match"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_boolean_d_sae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sonnet minor: ``bool`` is a Python ``int`` subclass
    (``isinstance(True, int) is True``), so a naive value-equality check
    alone could silently accept a corrupt boolean ``d_sae`` wherever it
    happens to numerically coincide with ``n_sae_features`` (``True == 1``,
    ``False == 0``). The strict ``type(loaded_d_sae) is not int`` guard
    rejects a bool outright, independent of what value it compares equal
    to."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            d_sae=True,  # bool, not a genuine int
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="d_sae/n_features does not match"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_wrong_loaded_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            model_name="gemma-2-9b",  # WRONG -- checkpoint config says gemma-2-2b
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="loaded SAE model_name"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_conflicting_direct_and_metadata_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            model_name="gemma-2-2b",
            metadata=SimpleNamespace(
                hook_name="blocks.8.hook_resid_post",
                model_name="gemma-2-9b",  # disagrees with the direct value
            ),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="model_name .* disagrees with"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_rejects_a_non_cpu_w_dec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU placement is contractual for this preflight (see the module
    docstring); asserted directly rather than trusted from the loader's own
    device kwarg. ``device="meta"`` needs no real accelerator hardware."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2), device="meta"),
        cfg=SimpleNamespace(d_in=2, metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post")),
    )
    with pytest.raises(ClusterFeatureContractError, match="must be a CPU tensor"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


def test_resolve_and_validate_sae_accepts_a_genuinely_matching_direct_release_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No false positive: direct release/model_name that genuinely match the
    resolver must not be rejected."""
    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            release=_SAE_RESOLVED.release,
            model_name="gemma-2-2b",
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    identity = ccp._resolve_and_validate_sae(
        metadata=_sae_metadata(),
        feature_directory=_sae_feature_directory(),
        n_sae_features=16384,
        sae_loader=lambda _resolved: sae,
    )
    assert identity.release == _SAE_RESOLVED.release


# --------------------------------------------------------------------------
# ``CheckpointSelectionContract``
# --------------------------------------------------------------------------


def _selection_contract_fixture(**overrides: Any) -> CheckpointSelectionContract:
    fields: dict[str, Any] = {
        "positive_selection_json": json.dumps(
            {"distribution": "geometric", "mean": 2.0}, sort_keys=True, separators=(",", ":")
        ),
        "negative_selection_json": json.dumps(
            {"distribution": "geometric", "mean": 1.0}, sort_keys=True, separators=(",", ":")
        ),
        "num_buckets": 2,
        "effective_positive_buckets": 1,
        "preselection_seed": 42,
        "selection_hash": _A_SHA,
        "selection_manifest_sha256": _B_SHA,
        "selection_shards_sha256": _C_SHA,
        "packed_row_fingerprint": _D_SHA,
    }
    fields.update(overrides)
    return CheckpointSelectionContract(**fields)


def test_checkpoint_selection_contract_canonical_and_provenance_dicts_pass_canonical_json_strict_domain() -> (
    None
):
    selection = _selection_contract_fixture(selection_manifest_file_sha256=_A_SHA)
    canonical = selection.canonical_dict()
    provenance = selection.provenance_dict()
    # Round-trips through B1's strict canonical-JSON domain without raising.
    cci._canonical_json(canonical)
    cci._canonical_json(provenance)
    assert canonical == {
        "positive_selection": {"distribution": "geometric", "mean": 2.0},
        "negative_selection": {"distribution": "geometric", "mean": 1.0},
        "num_buckets": 2,
        "effective_positive_buckets": 1,
        "preselection_seed": 42,
        "selection_hash": _A_SHA,
        "selection_manifest_sha256": _B_SHA,
        "selection_shards_sha256": _C_SHA,
        "packed_row_fingerprint": _D_SHA,
        "feature_delta": None,
    }
    assert provenance == {"selection_manifest_file_sha256": _A_SHA}


def test_checkpoint_selection_contract_file_sha_is_provenance_only_not_semantic() -> None:
    """``selection_manifest_file_sha256`` is ``compare=False`` — two contracts
    differing only in that field (e.g. the same manifest content re-derived
    from a byte-different-but-semantically-equal file) must still compare
    equal, mirroring ``CheckpointPin``'s own audit-vs-identity split.
    """
    with_hash = _selection_contract_fixture(selection_manifest_file_sha256=_A_SHA)
    without_hash = _selection_contract_fixture(selection_manifest_file_sha256=None)
    assert with_hash == without_hash


# --------------------------------------------------------------------------
# ``_resolve_selection_identity`` — trim-by-import (oracle's 9-case
# unsupported-training-contract parametrization) plus new raise-site
# coverage and one dead-branch documentation test.
# --------------------------------------------------------------------------


def _selection_user_config(**overrides: Any) -> dict[str, Any]:
    user_config: dict[str, Any] = {
        "feature_selection": {"distribution": "geometric", "mean": 2.0},
        "negative_selection": {"distribution": "geometric", "mean": 1.0},
        "num_buckets": 2,
        "seed": 42,
        "feature_delta": None,
        "feature_prediction": None,
    }
    user_config.update(overrides)
    return user_config


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing_negative", "requires negative_selection"),
        ("wrong_bucket_count", "num_buckets == 2"),
        ("raw_threshold", "only percentile-space"),
        ("threshold_without_gcg_k", "require explicit gcg_top_k"),
        ("gradient", "gradient-attribution"),
        ("delta_weights", "delta_weights disabled"),
        # Was ("feature_delta", "feature_delta=null"): the blanket
        # feature_delta=null requirement is gone. An UNSUPPORTED delta mode is
        # still rejected, which is what this case now pins.
        ("feature_delta_mode", "supports only feature_delta.mode"),
        # Was "feature_prediction=null": a head is now admissible, but only
        # for methods that never read it, and an UNDECLARED method set is
        # still rejected -- which is what this case now pins.
        ("feature_prediction", "declare enabled_methods"),
        ("self_conditioning", "self_conditioning=null/false"),
    ],
)
def test_clustered_selection_preflight_rejects_unsupported_training_contracts(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _selection_user_config()
    if case == "missing_negative":
        del user_config["negative_selection"]
    elif case == "wrong_bucket_count":
        user_config["num_buckets"] = 3
    elif case == "raw_threshold":
        user_config["feature_selection"] = {"threshold_type": "segment_topk", "threshold": 3}
    elif case == "threshold_without_gcg_k":
        user_config["feature_selection"] = {
            "threshold_type": "feature_percentile",
            "threshold": 0.9,
        }
    elif case == "gradient":
        user_config["feature_selection"] = {
            "distribution": "geometric",
            "mean": 2.0,
            "mode": "top_k_percentile",
            "gradient": {"mode": "future_half_split"},
        }
    elif case == "delta_weights":
        user_config["feature_selection"]["delta_weights"] = "absolute_percentile"
    elif case == "feature_delta_mode":
        user_config["feature_delta"] = {"mode": "prev_segment"}
    elif case == "feature_prediction":
        user_config["feature_prediction"] = {"enabled": True}
    elif case == "self_conditioning":
        user_config["self_conditioning"] = True
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)

    with pytest.raises(ClusterFeatureContractError, match=match):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_resolve_selection_identity_rejects_missing_user_config(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    with pytest.raises(ClusterFeatureContractError, match="checkpoint user_config is missing"):
        ccp._resolve_selection_identity(
            metadata={},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_resolve_selection_identity_rejects_non_mapping_feature_selection(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _selection_user_config()
    user_config["feature_selection"] = "not-a-mapping"
    with pytest.raises(ClusterFeatureContractError, match="requires feature_selection metadata"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_resolve_selection_identity_wraps_an_invalid_selection_config(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _selection_user_config()
    # Neither 'distribution' nor 'threshold_type' — parse_feature_selection's
    # own "both or neither -> error" ValueError, wrapped here.
    user_config["feature_selection"] = {}
    with pytest.raises(ClusterFeatureContractError, match="invalid checkpoint selection config"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_resolve_selection_identity_rejects_a_non_strict_int_seed(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _selection_user_config(seed="42")
    with pytest.raises(ClusterFeatureContractError, match="preselection seed must be a strict int"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


@pytest.mark.parametrize("disabled_looking_value", [0, 0.0])
def test_resolve_selection_identity_rejects_self_conditioning_zero_as_not_disabled(
    tmp_path: Path, disabled_looking_value: object
) -> None:
    """Codex combined-review finding 5 (MEDIUM) reproduction: ``value not in
    (None, False)`` is a value-EQUALITY test, and Python's ``0 == False`` /
    ``0.0 == False`` are both True — ``self_conditioning=0`` (or ``0.0``)
    silently passed as "disabled" without ever literally being the
    null/False sentinel this check means to require.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _selection_user_config()
    user_config["self_conditioning"] = disabled_looking_value
    with pytest.raises(ClusterFeatureContractError, match="self_conditioning=null/false"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_resolve_selection_identity_negative_disabled_raise_needs_a_bypassed_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead-branch documentation (same class of finding as B2a's own
    ``_statistics_identity`` domain-loop note): by the time
    ``parse_negative_selection`` is called here, ``negative_dict`` is always
    a genuine ``dict`` built from an already ``isinstance(..., Mapping)``-
    validated ``negative_raw`` — never Python ``None`` — because
    ``parse_negative_selection`` only ever returns ``None`` for ``cfg is
    None`` exactly (see ``steering/conditional/sampling.py``). The earlier
    Mapping check therefore already raises for any input that would make
    ``parse_negative_selection`` return ``None``, so ``if negative is None:
    raise ...`` can never fire via any real checkpoint metadata. Proven live
    here only by directly monkeypatching the parser to return ``None``,
    bypassing the earlier gate — regression-safety net for a future
    call-order refactor, not reachable production behavior today.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    monkeypatch.setattr(ccp, "parse_negative_selection", lambda _cfg: None)
    with pytest.raises(ClusterFeatureContractError, match="negative_selection is disabled"):
        ccp._resolve_selection_identity(
            metadata={"user_config": _selection_user_config()},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


# --------------------------------------------------------------------------
# ``resolve_selection_manifest_path``
# --------------------------------------------------------------------------


def test_selection_manifest_is_derived_from_checkpoint_selection_identity(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    positive = {"distribution": "geometric", "mean": 4.0, "std": 4.0}
    negative = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    metadata = {
        "user_config": {
            "feature_selection": positive,
            "negative_selection": negative,
            "num_buckets": 2,
            "seed": 17,
            "feature_delta": None,
            "feature_prediction": None,
        }
    }
    selection_hash = compute_selection_hash(
        positive_selection=positive,
        negative_selection=negative,
        delta_config=None,
        num_buckets=2,
        seed=17,
        ddp_rank=0,
        feature_mask=mask.canonical_dict(),
        feature_clusters=clusters.descriptor_dict(),
    )
    manifest = tmp_path / "packed" / "sel" / selection_hash / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    assert (
        resolve_selection_manifest_path(
            feature_dir=tmp_path,
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
        )
        == manifest.resolve()
    )

    duplicate = tmp_path / "other" / "sel" / selection_hash / "manifest.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("{}", encoding="utf-8")
    with pytest.raises(ClusterFeatureContractError, match="resolved 2 manifests"):
        resolve_selection_manifest_path(
            feature_dir=tmp_path,
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
        )


def test_resolve_selection_manifest_path_rejects_zero_matches(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    metadata = {"user_config": _selection_user_config()}
    with pytest.raises(ClusterFeatureContractError, match="resolved 0 manifests"):
        resolve_selection_manifest_path(
            feature_dir=tmp_path,
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
        )


def test_resolve_selection_manifest_path_rejects_a_symlinked_manifest(tmp_path: Path) -> None:
    """Codex combined-review finding 4 (MEDIUM) reproduction: an in-root
    ``sel/<hash>/manifest.json`` symlink to an external, unrelated matching
    artifact tree used to resolve to that outside target and be accepted as
    a genuine candidate."""
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    metadata = {"user_config": _selection_user_config()}
    identity = ccp._resolve_selection_identity(
        metadata=metadata,
        feature_mask=mask,
        feature_clusters=clusters,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
    )
    external_manifest = tmp_path / "outside" / "manifest.json"
    external_manifest.parent.mkdir(parents=True)
    external_manifest.write_text("{}", encoding="utf-8")

    in_root_manifest = tmp_path / "packed" / "sel" / identity.selection_hash / "manifest.json"
    in_root_manifest.parent.mkdir(parents=True)
    in_root_manifest.symlink_to(external_manifest)

    with pytest.raises(ClusterFeatureContractError, match="must not be a symlink"):
        resolve_selection_manifest_path(
            feature_dir=tmp_path,
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
        )


def test_resolve_selection_manifest_path_rejects_a_resolved_path_outside_the_feature_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment check (Codex combined-review finding 4, MEDIUM): even a
    non-symlink leaf whose layout matches ``sel/<hash>/manifest.json`` must
    still be rejected if it does not resolve inside ``feature_dir``.

    Dead-branch documentation (same class of finding as this module's other
    documented "kept for defense in depth" checks): ``Path.rglob(
    "manifest.json")`` does not, on this Python version, recurse into a
    symlinked directory at all without an explicit ``recurse_symlinks=True``
    (not passed) — a genuinely on-disk symlinked ANCESTOR directory is
    therefore never even discovered by the production call, so this
    specific escape shape cannot be reached via a real on-disk symlinked
    parent today (the LEAF-symlink case above remains fully reachable and
    is this finding's actual reproduction). Proven live here by
    monkeypatching discovery to yield one already-resolved, genuinely
    external candidate directly, isolating the containment check's own
    behavior from that discovery-mechanism detail — regression-safety net
    for a future discovery-mechanism change (e.g. ``recurse_symlinks=True``,
    or a switch to ``os.walk(followlinks=True)``), not reachable production
    behavior today.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    metadata = {"user_config": _selection_user_config()}
    identity = ccp._resolve_selection_identity(
        metadata=metadata,
        feature_mask=mask,
        feature_clusters=clusters,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
    )
    # Genuinely outside feature_dir=tmp_path -- a sibling of tmp_path, not a
    # subdirectory of it.
    external_root = tmp_path.parent / f"{tmp_path.name}_external"
    external_manifest = external_root / "sel" / identity.selection_hash / "manifest.json"
    external_manifest.parent.mkdir(parents=True)
    external_manifest.write_text("{}", encoding="utf-8")
    assert not external_manifest.is_symlink()

    root = tmp_path.resolve(strict=True)
    real_rglob = Path.rglob

    def rglob_with_an_escaped_candidate(self: Path, pattern: str):
        yield from real_rglob(self, pattern)
        if self == root and pattern == "manifest.json":
            yield external_manifest

    monkeypatch.setattr(Path, "rglob", rglob_with_an_escaped_candidate)

    with pytest.raises(ClusterFeatureContractError, match="resolved outside the feature root"):
        resolve_selection_manifest_path(
            feature_dir=tmp_path,
            metadata=metadata,
            feature_mask=mask,
            feature_clusters=clusters,
        )


# --------------------------------------------------------------------------
# ``_selection_manifest_semantic_sha256``
# --------------------------------------------------------------------------


def test_selection_manifest_semantic_hash_excludes_timestamp_and_cluster_path() -> None:
    first = {
        "checksum": _A_SHA,
        "kind": "packed_row",
        "created_at": "2026-07-18T12:00:00Z",
        "metadata": {
            "feature_clusters": {
                "source_path": "/workspace/clusters.yaml",
                "canonical": {"schema_version": 1, "clusters": [[1, 2]]},
            },
            "selection_shards": {"aggregate_sha256": _B_SHA},
        },
    }
    moved = json.loads(json.dumps(first))
    moved["created_at"] = "2026-07-19T09:30:00Z"
    moved["metadata"]["feature_clusters"]["source_path"] = "/root/clusters.yaml"

    assert ccp._selection_manifest_semantic_sha256(
        first
    ) == ccp._selection_manifest_semantic_sha256(moved)

    moved["metadata"]["feature_clusters"]["canonical"]["clusters"] = [[1, 3]]
    assert ccp._selection_manifest_semantic_sha256(
        first
    ) != ccp._selection_manifest_semantic_sha256(moved)


def test_selection_manifest_semantic_hash_excludes_only_derivable_validation() -> None:
    first = {
        "checksum": _A_SHA,
        "kind": "packed_row",
        "metadata": {
            "selection_shards": {"aggregate_sha256": _B_SHA},
            CLUSTERED_SELECTION_VALIDATION_KEY: {
                "aggregate_sha256": _C_SHA,
            },
        },
    }
    reissued = json.loads(json.dumps(first))
    reissued["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "aggregate_sha256"
    ] = _D_SHA

    assert ccp._selection_manifest_semantic_sha256(
        first
    ) == ccp._selection_manifest_semantic_sha256(reissued)

    reissued["metadata"]["selection_shards"]["aggregate_sha256"] = _D_SHA
    assert ccp._selection_manifest_semantic_sha256(
        first
    ) != ccp._selection_manifest_semantic_sha256(reissued)


def test_selection_manifest_semantic_hash_rejects_a_non_dict_json_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead-branch documentation: for this function's own ``Mapping[str,
    Any]`` parameter contract, B1's ``_canonical_json`` always either raises
    first (any non-``dict`` ``Mapping`` fails its ``json.dumps`` call, caught
    and re-raised as ``ClusterFeatureContractError`` before this function
    ever sees a result) or produces a JSON *object* string (a genuine
    ``dict`` always round-trips through ``json.loads`` back into a ``dict``)
    — so ``isinstance(semantic, dict)`` cannot fail for any real ``Mapping``
    input reaching this function. Proven live via a direct bypass of
    ``_canonical_json`` (returning a JSON *array* instead), not a real
    (unreachable) call shape.
    """
    monkeypatch.setattr(ccp, "_canonical_json", lambda _value: "[1,2,3]")
    with pytest.raises(ClusterFeatureContractError, match="selection manifest must be a mapping"):
        ccp._selection_manifest_semantic_sha256({"anything": "goes"})


# --------------------------------------------------------------------------
# ``_selection_contract`` — fixture builds every real on-disk artifact the
# function reads (packed-row shard + manifest, clustered selection shard +
# manifest) via the same production save helpers
# ``scripts/preprocess/precompute_selected_features.py`` uses, mirroring the
# oracle's own ``_write_full_preflight_fixture`` subset for this layer (that
# fixture's checkpoint/SAE/B3-orchestration parts are out of B2b's scope and
# stay deferred to B3).
# --------------------------------------------------------------------------

_SELECTION_MASK_IDS = (1, 2, 3, 4, 5, 6)
_SELECTION_PCT_CHECKSUM = "f" * 64


def _write_selection_fixture(
    tmp_path: Path,
    *,
    positive: dict[str, Any] | None = None,
    negative: dict[str, Any] | None = None,
    num_buckets: int = 2,
    seed: int = 42,
    legacy_splice_certificate: bool = False,
) -> tuple[
    dict[str, Any],
    Path,
    FeatureMask,
    ResolvedFeatureClusters,
    FeatureClusterConsolidation,
    str,
    Path,
]:
    positive = positive or {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    negative = negative or {"distribution": "geometric", "mean": 1.0, "std": 1.0}
    feature_delta = (
        {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
        if legacy_splice_certificate
        else None
    )
    mask = FeatureMask(FeatureMaskMode.whitelist, _SELECTION_MASK_IDS)
    clusters = _clusters(tmp_path / "clusters.yaml")
    consolidation = FeatureClusterConsolidation.build(clusters, mask)

    packed_dir = tmp_path / "features" / "packed"
    packed_dir.mkdir(parents=True)
    packed_path = save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 4], dtype=t.int32),
            feature_ids=t.tensor([1, 2, 4, 5], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180, 200, 90], dtype=t.uint8),
        ),
        packed_dir / "shard_00000",
    )
    write_manifest(
        packed_dir,
        "packed-fixture",
        OverlayKind.PACKED_ROW,
        metadata={"pct_checksum": _SELECTION_PCT_CHECKSUM},
        base_shard_paths=[packed_path],
    )

    selection_hash = compute_selection_hash(
        positive_selection=positive,
        negative_selection=negative,
        delta_config=feature_delta,
        num_buckets=num_buckets,
        seed=seed,
        ddp_rank=0,
        feature_mask=mask.canonical_dict(),
        feature_clusters=clusters.descriptor_dict(),
    )
    selection_dir = packed_dir / "sel" / selection_hash
    selection_dir.mkdir(parents=True)
    selection_path = save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([1, 4], dtype=t.uint16),
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            # 3 (not 5): with clusters=((1,2),(4,5)), id 5 is CONSOLIDATED
            # AWAY into representative 4 (see _clusters()'s default). 3 is
            # unclustered and therefore its own representative -- the only
            # valid choice here now that Codex combined-review finding 1
            # makes _selection_contract call the canonical
            # FeatureClusterConsolidation.assert_selection_shard validator,
            # which rejects consolidated-away IDs on both the positive AND
            # negative side. This fixture was never actually semantically
            # validated before that fix landed.
            neg_fids=t.tensor([3], dtype=t.uint16),
        ),
        selection_dir / "shard_00000",
    )
    cluster_metadata = consolidation.metadata_dict()
    cluster_metadata["status"] = "complete"
    selection_integrity = build_clustered_selection_shard_integrity(
        [selection_path]
    )
    packed_proof = resolve_packed_row_fingerprint_strict(packed_dir)
    selection_metadata: dict[str, Any] = {
        "feature_mask": mask.canonical_dict(),
        "feature_clusters": cluster_metadata,
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(packed_dir)
        },
        "selection_shards": selection_integrity,
    }
    if legacy_splice_certificate:
        content_digest = selection_integrity["shards"][0]["content_sha256"]
        splice_stats = SpliceSelectionStats(
            total_packed_segments=1,
            total_sidecar_splice_positions_referenced=0,
            total_delta_positions_selected=0,
        )
        selection_metadata["splice_counterfactual"] = {
            "schema_version": (
                CLUSTERED_SPLICE_COUNTERFACTUAL_MANIFEST_SCHEMA_VERSION
            ),
            "status": "complete",
            CLUSTER_DELTA_SEMANTICS_KEY: (
                CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
            ),
            "total_packed_segments_selected": 1,
            "total_sidecar_splice_positions_referenced": 0,
            "total_delta_positions_selected": 0,
        }
        selection_metadata[SPLICE_RESUME_METADATA_KEY] = (
            build_splice_resume_metadata(
                stats_by_stem={"shard_00000": splice_stats},
                selection_integrity=selection_integrity,
                storage_validated_by_stem={
                    "shard_00000": content_digest,
                },
                validated_max_features_per_segment=3,
                packed_rows_hash=packed_proof.aggregate_sha256,
                effective_buckets=num_buckets - 1,
                status="complete",
            )
        )
        selection_metadata[REMOTE_LOADER_KEY] = (
            build_remote_loader_certificate(
                selection_checksum=selection_hash,
                selection_inventory_aggregate=selection_integrity[
                    "aggregate_sha256"
                ],
                packed_row_fingerprint=packed_proof.aggregate_sha256,
                max_features_per_segment=3,
                cluster_delta_semantics=(
                    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
                ),
            )
        )
    else:
        selection_metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = (
            build_clustered_selection_validation(
                selection_checksum=selection_hash,
                selection_integrity=selection_integrity,
                packed_row_fingerprint=packed_proof.aggregate_sha256,
                packed_content_by_name={
                    entry.name.removesuffix(".safetensors.zst"): (
                        entry.content_sha256
                    )
                    for entry in packed_proof.files
                },
                feature_clusters_descriptor=clusters.descriptor_dict(),
                feature_delta=None,
                effective_buckets=num_buckets - 1,
                segment_counts_by_stem={"shard_00000": 1},
                max_features_by_stem={"shard_00000": 3},
            )
        )
    write_manifest(
        selection_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata=selection_metadata,
        base_shard_paths=[packed_path],
        cache_dir=packed_dir,
    )

    metadata = {
        "user_config": {
            "feature_selection": positive,
            "negative_selection": negative,
            "num_buckets": num_buckets,
            "seed": seed,
            "feature_delta": feature_delta,
            "feature_prediction": None,
        }
    }
    manifest_path = selection_dir / "manifest.json"
    return (
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        _SELECTION_PCT_CHECKSUM,
        selection_path,
    )


def _call_selection_contract(
    metadata: dict[str, Any],
    manifest_path: Path,
    mask: FeatureMask,
    clusters: ResolvedFeatureClusters,
    consolidation: FeatureClusterConsolidation,
    percentile_checksum: str,
    *,
    force_cluster_shard_revalidation: bool = False,
) -> CheckpointSelectionContract:
    return ccp._selection_contract(
        metadata=metadata,
        manifest_path=manifest_path,
        feature_mask=mask,
        feature_clusters=clusters,
        consolidation=consolidation,
        percentile_checksum=percentile_checksum,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
        force_cluster_shard_revalidation=(
            force_cluster_shard_revalidation
        ),
    )


@pytest.mark.parametrize(
    "field",
    ["force_cluster_shard_revalidation", "allow_packed_cache_write"],
)
def test_selection_contract_rejects_bool_lookalikes(
    tmp_path: Path,
    field: str,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    kwargs: dict[str, Any] = {
        "metadata": metadata,
        "manifest_path": manifest_path,
        "feature_mask": mask,
        "feature_clusters": clusters,
        "consolidation": consolidation,
        "percentile_checksum": percentile_checksum,
        "gcg_top_k_pos": None,
        "gcg_top_k_neg": None,
        field: 1,
    }
    with pytest.raises(
        ClusterFeatureContractError,
        match=f"{field} must be a bool",
    ):
        ccp._selection_contract(**kwargs)


def _reissue_fixture_validation_certificate(
    manifest_path: Path,
    clusters: ResolvedFeatureClusters,
    *,
    segment_count: int = 1,
    max_features_per_segment: int = 3,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    packed_dir = manifest_path.parent.parent.parent
    packed_proof = resolve_packed_row_fingerprint_strict(
        packed_dir,
        force=True,
    )
    metadata["packed_rows"]["packed_row_fingerprint"] = (
        packed_proof.aggregate_sha256
    )
    metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = (
        build_clustered_selection_validation(
            selection_checksum=manifest["checksum"],
            selection_integrity=metadata["selection_shards"],
            packed_row_fingerprint=packed_proof.aggregate_sha256,
            packed_content_by_name={
                entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
                for entry in packed_proof.files
            },
            feature_clusters_descriptor=clusters.descriptor_dict(),
            feature_delta=None,
            effective_buckets=1,
            segment_counts_by_stem={"shard_00000": segment_count},
            max_features_by_stem={
                "shard_00000": max_features_per_segment,
            },
        )
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )


def test_selection_contract_certified_warm_path_skips_all_tensor_body_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )

    def forbidden(*_args: object, **_kwargs: object):
        raise AssertionError("certified warm path opened a tensor or packed body")

    monkeypatch.setattr(ccp, "load_packed_row_shard_bytes", forbidden)
    monkeypatch.setattr(ccp, "load_selection_shard", forbidden)
    monkeypatch.setattr(splice_module, "_file_fingerprint", forbidden)
    selection = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
    )
    assert selection.selection_shards_sha256


def test_selection_contract_legacy_splice_v3_warm_path_matches_forced_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The motivating legacy producer proof gets the same certified fast path."""

    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(
            tmp_path,
            legacy_splice_certificate=True,
        )
    )
    integrity_calls = 0
    real_integrity_check = ccp.assert_clustered_selection_shard_integrity

    def recording_integrity(*args: Any, **kwargs: Any) -> Any:
        nonlocal integrity_calls
        integrity_calls += 1
        return real_integrity_check(*args, **kwargs)

    def forbidden(*_args: object, **_kwargs: object):
        raise AssertionError("legacy certified warm path opened a tensor body")

    with monkeypatch.context() as warm_patch:
        warm_patch.setattr(
            ccp,
            "assert_clustered_selection_shard_integrity",
            recording_integrity,
        )
        warm_patch.setattr(ccp, "load_packed_row_shard_bytes", forbidden)
        warm_patch.setattr(ccp, "load_selection_shard", forbidden)
        warm_patch.setattr(splice_module, "_file_fingerprint", forbidden)
        fast = _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )

    forced = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
        force_cluster_shard_revalidation=True,
    )

    assert integrity_calls == 1
    assert forced == fast
    assert forced.canonical_dict() == fast.canonical_dict()
    assert forced.provenance_dict() == fast.provenance_dict()


def test_legacy_splice_v3_forced_path_rejects_forged_global_width(
    tmp_path: Path,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(
            tmp_path,
            legacy_splice_certificate=True,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_metadata = manifest["metadata"]
    integrity = selection_metadata["selection_shards"]
    resume = selection_metadata[SPLICE_RESUME_METADATA_KEY]
    packed_fingerprint = selection_metadata["packed_rows"][
        "packed_row_fingerprint"
    ]
    effective_buckets = resume["output_validation"]["effective_buckets"]
    stats, _complete, storage_validated, actual_width = (
        parse_splice_resume_metadata(
            resume,
            selection_integrity=integrity,
            packed_rows_hash=packed_fingerprint,
            effective_buckets=effective_buckets,
            manifest_status="complete",
            surface="legacy forged-width fixture",
        )
    )
    assert actual_width == 3
    forged_width = 99
    selection_metadata[SPLICE_RESUME_METADATA_KEY] = (
        build_splice_resume_metadata(
            stats_by_stem=stats,
            selection_integrity=integrity,
            storage_validated_by_stem=storage_validated,
            validated_max_features_per_segment=forged_width,
            packed_rows_hash=packed_fingerprint,
            effective_buckets=effective_buckets,
            status="complete",
        )
    )
    selection_metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum=manifest["checksum"],
        selection_inventory_aggregate=integrity["aggregate_sha256"],
        packed_row_fingerprint=packed_fingerprint,
        max_features_per_segment=forged_width,
        cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="exact packed-shard domain and global maximum selection width",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_real_clustered_producer_artifact_resolves_fast_and_forced_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the actual producer, then consume its published artifact end to end."""

    env = build_interleaved_env(tmp_path, n_shards=2)
    cluster_path = write_cluster_yaml(tmp_path)
    run_selection_main(
        env,
        num_workers=1,
        cache_shards=2,
        sidecar_shards=4,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=cluster_path,
        monkeypatch=monkeypatch,
    )
    selection_root = Path(env["packed_row_dir"]) / "sel"
    selection_dirs = [path for path in selection_root.iterdir() if path.is_dir()]
    assert len(selection_dirs) == 1
    manifest_path = selection_dirs[0] / "manifest.json"
    published = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert published["metadata"][CLUSTERED_SELECTION_VALIDATION_KEY][
        "status"
    ] == "complete"
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(150)))
    clusters = resolve_feature_clusters(
        {"feature_clusters": {"path": cluster_path}},
        4096,
    )
    assert clusters is not None
    consolidation = FeatureClusterConsolidation.build(clusters, mask)
    metadata = {
        "user_config": {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "min_k": 0,
                "max_k": 49,
            },
            "num_buckets": 2,
            "seed": 42,
            "feature_delta": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            "feature_prediction": None,
        }
    }

    fast = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        env["pct_checksum"],
    )
    forced = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        env["pct_checksum"],
        force_cluster_shard_revalidation=True,
    )

    assert forced == fast
    assert forced.canonical_dict() == fast.canonical_dict()
    assert forced.provenance_dict() == fast.provenance_dict()


def test_selection_contract_forced_path_matches_fast_oracle_and_runs_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    fast = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
    )
    real_packed_load = ccp.load_packed_row_shard_bytes
    real_selection_load = ccp.load_selection_shard
    calls = {"packed": 0, "selection": 0}

    def packed_load(compressed: bytes):
        calls["packed"] += 1
        return real_packed_load(compressed)

    def selection_load(path: Path):
        calls["selection"] += 1
        return real_selection_load(path)

    monkeypatch.setattr(ccp, "load_packed_row_shard_bytes", packed_load)
    monkeypatch.setattr(ccp, "load_selection_shard", selection_load)
    forced = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
        force_cluster_shard_revalidation=True,
    )
    assert forced == fast
    assert forced.canonical_dict() == fast.canonical_dict()
    assert forced.provenance_dict() == fast.provenance_dict()
    assert calls == {"packed": 1, "selection": 1}


def test_selection_contract_forced_path_rejects_certified_width_disagreement(
    tmp_path: Path,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _reissue_fixture_validation_certificate(
        manifest_path,
        clusters,
        max_features_per_segment=99,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="producer-certificate width mismatch",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_forced_width_helper_rejects_positive_negative_segment_disagreement() -> None:
    malformed = SelectionShard(
        sel_offsets=t.tensor([0, 0, 0], dtype=t.int32),
        sel_fids=t.empty(0, dtype=t.uint16),
        sel_buckets=t.empty(0, dtype=t.uint8),
        sel_pcts=t.empty(0, dtype=t.uint8),
        neg_offsets=t.tensor([0, 0], dtype=t.int32),
        neg_fids=t.empty(0, dtype=t.uint16),
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="different segment counts",
    ):
        ccp._selection_shard_max_features_per_segment(malformed)


def test_selection_contract_wraps_strict_packed_final_fence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )

    def reject(_resolution: object) -> None:
        raise splice_module.SpliceProvenanceError("packed proof became stale")

    monkeypatch.setattr(
        ccp,
        "assert_packed_row_fingerprint_resolution_current",
        reject,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="packed proof became stale",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )


def test_selection_contract_forced_path_rejects_new_uncertified_packed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    (packed_path,) = list(packed_dir.glob("*.safetensors.zst"))
    real_resolve = ccp.resolve_packed_row_fingerprint_strict
    added = False

    def add_before_strict_resolve(*args: Any, **kwargs: Any):
        nonlocal added
        if not added:
            shutil.copyfile(
                packed_path,
                packed_dir / "shard_99999.safetensors.zst",
            )
            added = True
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        ccp,
        "resolve_packed_row_fingerprint_strict",
        add_before_strict_resolve,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="no selection shard",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_selection_contract_forced_path_runs_canonical_semantic_validator(
    tmp_path: Path,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([1, 5], dtype=t.uint16),
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([3], dtype=t.uint16),
        ),
        shard_path,
    )
    _mutate_manifest(
        manifest_path,
        lambda manifest: manifest["metadata"].__setitem__(
            "selection_shards",
            build_clustered_selection_shard_integrity([shard_path]),
        ),
    )
    _reissue_fixture_validation_certificate(manifest_path, clusters)

    with pytest.raises(
        ClusterFeatureContractError,
        match="consolidated-away feature ID 5",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_selection_contract_forced_path_requires_validator_callback_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    real_resolve = ccp.resolve_packed_row_fingerprint_strict

    def omit_body_validator(*args: Any, **kwargs: Any):
        kwargs["body_validator"] = None
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(
        ccp,
        "resolve_packed_row_fingerprint_strict",
        omit_body_validator,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="did not reproduce",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_selection_contract_rejects_producer_authority_packed_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    real_resolve = ccp.resolve_clustered_selection_validation

    def disagree(*args: Any, **kwargs: Any):
        authority = real_resolve(*args, **kwargs)
        return replace(authority, packed_row_fingerprint="0" * 64)

    monkeypatch.setattr(
        ccp,
        "resolve_clustered_selection_validation",
        disagree,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="producer validation packed-row fingerprint",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )


def test_selection_contract_wraps_producer_packed_inventory_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise FeatureClustersError("injected packed inventory disagreement")

    monkeypatch.setattr(ccp, "assert_validation_packed_inventory", reject)
    with pytest.raises(
        ClusterFeatureContractError,
        match="injected packed inventory disagreement",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )


def test_selection_contract_accepts_optional_cache_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    cache_path = packed_dir / ".packed-row-fingerprint-cache.json"
    cache_path.unlink()

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected optional cache write failure")

    monkeypatch.setattr(splice_module, "atomic_json_dump", fail_write)
    contract = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
    )
    assert contract is not None
    assert not cache_path.exists()


def test_selection_contract_rechecks_exact_inventory_after_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    cache_path = packed_dir / ".packed-row-fingerprint-cache.json"
    cache_path.unlink()
    real_store = splice_module._store_file_fingerprint_cache

    def store_then_mutate(*args: Any, **kwargs: Any) -> None:
        real_store(*args, **kwargs)
        (manifest_path.parent / "shard_99999.safetensors.zst").write_bytes(
            shard.read_bytes()
        )

    monkeypatch.setattr(
        splice_module,
        "_store_file_fingerprint_cache",
        store_then_mutate,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="selection shard inventory changed",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )
    assert cache_path.is_file()


def test_selection_contract_rechecks_packed_inventory_after_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    (packed_path,) = list(packed_dir.glob("*.safetensors.zst"))
    cache_path = packed_dir / ".packed-row-fingerprint-cache.json"
    cache_path.unlink()
    real_store = splice_module._store_file_fingerprint_cache

    def store_then_mutate(*args: Any, **kwargs: Any) -> None:
        real_store(*args, **kwargs)
        shutil.copyfile(
            packed_path,
            packed_dir / "shard_99999.safetensors.zst",
        )

    monkeypatch.setattr(
        splice_module,
        "_store_file_fingerprint_cache",
        store_then_mutate,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="packed-row shard inventory changed",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )
    assert cache_path.is_file()


def test_selection_contract_rechecks_manifest_after_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    cache_path = packed_dir / ".packed-row-fingerprint-cache.json"
    cache_path.unlink()
    original = manifest_path.read_bytes()
    mutated = original.replace(b'"checksum"', b'"ch3cksum"', 1)
    assert len(mutated) == len(original)
    real_store = splice_module._store_file_fingerprint_cache

    def store_then_mutate(*args: Any, **kwargs: Any) -> None:
        real_store(*args, **kwargs)
        manifest_path.write_bytes(mutated)

    monkeypatch.setattr(
        splice_module,
        "_store_file_fingerprint_cache",
        store_then_mutate,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="pinned contract input changed after hashing",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )
    assert cache_path.is_file()


def test_selection_contract_missing_producer_authority_fails_before_body_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(
        manifest_path,
        lambda manifest: manifest["metadata"].pop(
            CLUSTERED_SELECTION_VALIDATION_KEY
        ),
    )
    monkeypatch.setattr(
        ccp,
        "assert_clustered_selection_shard_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "missing authority reached selection body hashing"
        ),
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="not eligible for the legacy splice-v3 proof",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
        )


def _mutate_manifest(manifest_path: Path, mutate: Any) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_selection_contract_happy_path_binds_manifest_and_shard_provenance(
    tmp_path: Path,
) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    selection = _call_selection_contract(
        metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
    )
    assert selection.num_buckets == 2
    assert selection.preselection_seed == 42
    assert selection.effective_positive_buckets == 1
    assert is_sha256_hex(selection.selection_hash)
    assert is_sha256_hex(selection.selection_manifest_sha256)
    assert is_sha256_hex(selection.selection_shards_sha256)
    assert is_sha256_hex(selection.packed_row_fingerprint)
    assert (
        selection.selection_manifest_file_sha256
        == cci.stream_hash_file_stable(manifest_path).sha256
    )


def test_selection_contract_matches_frozen_origin_main_full_validator_oracle(
    tmp_path: Path,
) -> None:
    """Freeze the pre-PR full validator, independently of fast/forced parity.

    These values were emitted by the identical fixture from exported
    ``origin/main`` at ``ce039ac0d``. The raw manifest file digest is
    intentionally excluded: adding the derivable certificate changes audit
    provenance bytes while the semantic selection contract must stay exact.
    """

    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    selection = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
    )
    assert selection.canonical_dict() == {
        "positive_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 1.0,
            "std": 1.0,
        },
        "num_buckets": 2,
        "effective_positive_buckets": 1,
        "preselection_seed": 42,
        "selection_hash": (
            "be3f92dca0e73aad9afde20bbf42867160be0121c77f4167501cd28bc20c1a7e"
        ),
        "selection_manifest_sha256": (
            "5a5aacfc8ab2ff209bfd3f10a9c2cd037d9d534a54816e48c0c3bb441dff1823"
        ),
        "selection_shards_sha256": (
            "30d6702ecb9a63b36b14117d9b6417a30bc65dc47be594e468d71211871c7bc9"
        ),
        "packed_row_fingerprint": (
            "69e439647533c15bea9d11b8568ec3dc619b5b5dee5f5a7bc47921b5b0dbcf59"
        ),
        "feature_delta": None,
    }


def test_selection_contract_rejects_checksum_mismatch(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(manifest_path, lambda m: m.__setitem__("checksum", "0" * 64))
    with pytest.raises(ClusterFeatureContractError, match="checksum does not match"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_missing_manifest_metadata(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(manifest_path, lambda m: m.__setitem__("metadata", None))
    with pytest.raises(ClusterFeatureContractError, match="selection manifest metadata is missing"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_feature_mask_mismatch(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__("feature_mask", {"mode": "whitelist", "ids": [9]}),
    )
    with pytest.raises(
        ClusterFeatureContractError, match="feature mask does not match the checkpoint/config"
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_wraps_a_feature_clusters_metadata_error(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"]["feature_clusters"].__setitem__("status", "in_progress"),
    )
    with pytest.raises(ClusterFeatureContractError, match="status must be one of"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_a_tampered_selection_shard(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    with shard_path.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ClusterFeatureContractError, match="content fingerprint mismatch"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_missing_shard_aggregate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dead-branch documentation (same class of finding as B2a's own
    ``_statistics_identity`` domain-loop note, and this file's
    ``_resolve_selection_identity``/``_selection_manifest_semantic_sha256``
    findings above): ``assert_clustered_selection_shard_integrity``, called
    immediately before this check, already forces
    ``manifest_metadata["selection_shards"]["aggregate_sha256"]`` to equal a
    freshly recomputed, genuine SHA-256 hex digest whenever it *succeeds* —
    its own ``actual == expected`` dict-equality check compares the full
    ``{schema_version, shards, aggregate_sha256}`` mapping against one
    ``build_clustered_selection_shard_integrity`` recomputes from the real
    on-disk shard bytes, which always produces a canonical lowercase-hex
    ``_sha256_json(...)`` value for that field. Any manifest whose aggregate
    is malformed, absent, or simply stale therefore already fails that
    earlier call first — as "content fingerprint mismatch" (or a structural
    "must contain exactly [...]" shape error), not this message — before
    ``_selection_contract``'s own ``is_sha256_hex`` check is ever reached.
    Proven live only by bypassing the earlier call.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    monkeypatch.setattr(ccp, "assert_clustered_selection_shard_integrity", lambda *a, **k: None)
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__(
            "selection_shards", {"aggregate_sha256": "not-a-sha256"}
        ),
    )
    with pytest.raises(
        ClusterFeatureContractError, match="missing its selected-shard aggregate identity"
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_missing_packed_row_provenance(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__("packed_rows", {"packed_row_fingerprint": "too-short"}),
    )
    with pytest.raises(ClusterFeatureContractError, match="missing packed-row provenance"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_a_missing_parent_packed_row_manifest(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_manifest_path = manifest_path.parent.parent.parent / "manifest.json"
    assert packed_manifest_path.is_file()
    packed_manifest_path.unlink()
    with pytest.raises(ClusterFeatureContractError, match="missing a valid packed-row manifest"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_percentile_provenance_mismatch(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, _percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="percentile provenance does not match feature statistics",
    ):
        _call_selection_contract(metadata, manifest_path, mask, clusters, consolidation, "0" * 64)


def test_selection_contract_rejects_a_packed_row_fingerprint_mismatch(tmp_path: Path) -> None:
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"]["packed_rows"].__setitem__("packed_row_fingerprint", "0" * 64),
    )
    with pytest.raises(
        ClusterFeatureContractError, match="packed-row fingerprint does not match current bytes"
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_a_consolidated_away_feature_id_via_coordinated_re_signing(
    tmp_path: Path,
) -> None:
    """Codex combined-review finding 1 (HIGH), reproduction #1: replace a
    selection shard with a validly-formed but semantically-invalid shard (one
    containing a consolidated-away feature ID) and re-sign the manifest's own
    shard-integrity inventory/aggregate to match the new bytes exactly. This
    passes byte-level ``assert_clustered_selection_shard_integrity``
    self-consistency (declared inventory and on-disk bytes agree), but it
    cannot forge the producer-authority certificate. The normal certified
    path therefore fails before tensor decompression.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    # clusters=((1,2),(4,5)) (see _clusters()'s default): id 5 is
    # consolidated away into representative 4, so it must never appear as a
    # selected positive feature ID.
    save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([1, 5], dtype=t.uint16),  # 5 is consolidated-away
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([3], dtype=t.uint16),
        ),
        shard_path,
    )
    # Coordinated re-signing: recompute the manifest's OWN shard-integrity
    # inventory/aggregate from the tampered bytes so the earlier byte-level
    # self-consistency check (assert_clustered_selection_shard_integrity)
    # passes too -- exactly the attack Codex reproduced.
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__(
            "selection_shards",
            build_clustered_selection_shard_integrity([shard_path]),
        ),
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="selection_inventory_aggregate",
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_rejects_incomplete_packed_selection_shard_coverage(
    tmp_path: Path,
) -> None:
    """Codex combined-review finding 1 (HIGH), reproduction #2: add another
    packed-row shard and re-sign ``packed_row_fingerprint`` to match the new
    packed_dir state, while leaving the selection shard set unchanged (now
    covering only part of the packed domain). Prior to this fix,
    ``_selection_contract`` never cross-checked the selection shard set
    against the packed shard set at all -- despite the producer
    (``precompute_selected_features.py``) requiring exact stem equality at
    write time and the dataloader requiring it again at read time.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 2], dtype=t.int32),
            feature_ids=t.tensor([1, 2], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180], dtype=t.uint8),
        ),
        packed_dir / "shard_00001",
    )
    # Coordinated re-signing: recompute packed_row_fingerprint from the NEW
    # (2-shard) packed_dir state so the existing fingerprint-mismatch check
    # (below, unchanged) does not by itself catch the incomplete coverage --
    # exactly the attack Codex reproduced.
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"]["packed_rows"].__setitem__(
            "packed_row_fingerprint", packed_row_fingerprint(packed_dir)
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="must match packed shard stems exactly"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_detects_a_real_equal_size_manifest_swap_around_the_fused_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constraint 3's swap-race regression (the module docstring's dedicated
    section): same proof pattern as B2a's checkpoint-metadata regression,
    now exercised end-to-end through ``_selection_contract`` for the
    selection manifest — equal-size A->B swap staged immediately before
    ``open()``, restored to fresh A-content immediately after.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    resolved = manifest_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, resolved)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )
    assert swapped["done"]


def test_selection_contract_detects_a_real_equal_size_manifest_swap_around_the_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to the fused-hash regression above: the swap targets the
    *second* open (the payload read inside ``_load_json_mapping``, after the
    fd-bound hash already closed) — caught via content-hash verification
    against the digest from the first open.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    resolved = manifest_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            f"^selection manifest changed while it was being loaded: {re.escape(str(resolved))}$"
        ),
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )
    assert open_count["n"] >= 2


# --------------------------------------------------------------------------
# ``_selection_contract`` — Codex combined-review finding 3 (HIGH): the
# artifact set was not validated as one snapshot (mixed-snapshot races).
# --------------------------------------------------------------------------


def test_selection_contract_detects_a_late_selection_manifest_mutation_after_the_early_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex's exact reproduction: a mutation staged AFTER the selection
    manifest's own hash+read, but before the rest of the transaction
    finishes, used to go completely undetected — the cold recheck ran
    immediately after the manifest loaded, not after the last read that
    still depends on it. Simulated by mutating the manifest's on-disk bytes
    (in place, equal size, so no earlier size-derived check trips first) as
    a side effect of the LAST production call this function makes before
    its (now-moved-to-the-end) cold-check batch.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    resolved = manifest_path.resolve()
    original_bytes = resolved.read_bytes()
    mutated_bytes = original_bytes.replace(b'"checksum"', b'"ch3cksum"', 1)
    assert mutated_bytes != original_bytes
    assert len(mutated_bytes) == len(original_bytes)

    real_packed_resolver = ccp.resolve_packed_row_fingerprint_strict

    def mutating_packed_resolver(*args: Any, **kwargs: Any):
        result = real_packed_resolver(*args, **kwargs)
        resolved.write_bytes(mutated_bytes)
        return result

    monkeypatch.setattr(
        ccp,
        "resolve_packed_row_fingerprint_strict",
        mutating_packed_resolver,
    )

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^pinned contract input changed after hashing: {re.escape(str(resolved))}$",
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_detects_a_real_equal_size_packed_manifest_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packed-row manifest is now routed through the same digest-bound
    ``_load_json_mapping`` the selection manifest already used, not
    ``overlay_io.read_manifest``'s unbound pathname-check-then-separate-open.
    Same swap-race proof pattern as the selection-manifest regressions
    above (equal-size A->B swap staged immediately before ``open()``,
    restored to fresh A-content immediately after), now for
    ``packed_dir/manifest.json``.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_manifest_path = manifest_path.parent.parent.parent / "manifest.json"
    resolved = packed_manifest_path.resolve()
    a_bytes = resolved.read_bytes()
    b_bytes = b"X" * len(a_bytes)
    b_source = tmp_path / ".staged_packed_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_packed_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, resolved)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )
    assert swapped["done"]


def test_selection_contract_rejects_an_actual_packed_shard_byte_mutation(
    tmp_path: Path,
) -> None:
    """Finding 7 (MEDIUM): exercise a REAL packed-row shard byte mutation,
    not just a manifest-value edit. ``test_selection_contract_rejects_a_
    packed_row_fingerprint_mismatch`` above only ever mutates the caller's
    EXPECTED manifest value; it never proves ``packed_row_fingerprint``'s
    own recomputation actually re-reads genuine on-disk bytes. This mutates
    a real packed shard directly and relies on nothing but that
    recomputation (mirroring the shard-tamper test's existing
    selection-shard counterpart, ``test_selection_contract_rejects_a_
    tampered_selection_shard``, on the packed side).
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    (shard_path,) = list(packed_dir.glob("*.safetensors.zst"))
    original_bytes = shard_path.read_bytes()
    with shard_path.open("ab") as handle:
        handle.write(b"\x00")
    assert shard_path.read_bytes() != original_bytes
    with pytest.raises(ClusterFeatureContractError):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


# --------------------------------------------------------------------------
# ``_selection_contract`` — zero-copy production-parity rewrite (owner-
# approved architecture change, superseding the Codex convergence-pass
# residual 1 temp-materialization fix above; see the module docstring's
# "OWNER SCOPE DECISION" / "Accepted threat-model boundary" sections for the
# full ledger). Shard bodies are validated BY PATH again, exactly as
# ``dataloader._validate_generic_selection_dir`` reads these identical
# files, with a stat-fence (``_shard_stat_signatures``/
# ``_assert_shard_stat_signatures_unchanged``) bracketing every canonical
# read this function performs.
# --------------------------------------------------------------------------


def test_selection_contract_rejects_a_selection_shard_a_to_b_to_a_schedule_via_the_stat_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex convergence-pass residual 1 (HIGH), exact reproduction against
    the reverted, production-parity (no-copy) implementation: the
    byte-integrity check reads selection shard A -- a consolidated-away-ID
    shard (mirroring
    ``test_selection_contract_rejects_a_consolidated_away_feature_id_via_coordinated_re_signing``),
    coordinately re-signed so the manifest's own byte-integrity aggregate
    matches it exactly -- and the per-shard semantic loader
    (``load_selection_shard``) independently reopens the SAME live pathname
    a second time. That second reopen is production parity with the
    dataloader, not a bug (``dataloader._validate_generic_selection_dir``
    does the identical two separate opens); Codex's original schedule swaps
    in valid B for exactly that second read, then restores A immediately
    after, before ``_selection_contract`` returns.

    B is the fixture's genuinely valid shard -- deliberately crafted so a
    validator that only ever sees B would find nothing wrong, even though A
    (what is actually on disk when this function returns) contains a
    consolidated-away feature ID the semantic validator never inspected.
    Without the shard stat-fence this PR adds, this schedule would return
    successfully, certifying bytes it never actually validated (Codex's
    exact pass-3 "stale-before-return" finding). The stat-fence's final
    recheck -- after the last shard-body read, mirroring the two manifest
    digests' own final cold checks -- catches it here: a rename-based
    restore always lands a fresh inode, so the restored pathname's stat
    signature differs from what was captured before any of this began, and
    ``_assert_shard_stat_signatures_unchanged`` rejects it.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    resolved = shard_path.resolve()
    b_bytes = resolved.read_bytes()  # the fixture's genuinely valid shard

    # A: consolidated-away ID 5 (clusters=((1,2),(4,5)); 4 is 5's
    # representative), coordinately re-signed so the manifest's OWN
    # byte-integrity aggregate matches these new bytes exactly -- mirrors
    # test_selection_contract_rejects_a_consolidated_away_feature_id_via_coordinated_re_signing.
    save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([1, 5], dtype=t.uint16),  # 5 is consolidated-away
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([3], dtype=t.uint16),
        ),
        shard_path,
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__(
            "selection_shards",
            build_clustered_selection_shard_integrity([shard_path]),
        ),
    )
    _reissue_fixture_validation_certificate(manifest_path, clusters)
    a_bytes = resolved.read_bytes()
    assert a_bytes != b_bytes
    b_source = tmp_path / ".staged_b_valid"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = Path.open
    # Production parity means the selection shard is now read BY PATH more
    # than once per call (the byte-integrity check, then the per-shard
    # semantic loader) -- target the SECOND open specifically, so this test
    # exercises the stat-fence rather than the byte-integrity primitive's
    # own pre-existing fd-bound before/after check (which would reject a
    # swap staged around the FIRST open on its own, unrelated to this PR).
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            "^a selection/packed shard changed while this contract was "
            f"validating it: {re.escape(str(resolved))}$"
        ),
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )
    assert open_count["n"] >= 2
    assert resolved.read_bytes() == a_bytes


def test_selection_contract_detects_a_real_equal_size_packed_shard_swap_around_a_canonical_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to the selection-shard stat-fence regression above, now for
    a packed shard body. Production parity means the packed shard is read
    BY PATH more than once within one ``_selection_contract`` call
    (``load_packed_row_shard`` in the per-stem loop, then
    ``packed_row_fingerprint``'s own internal read) -- exactly as
    ``dataloader._validate_generic_selection_dir`` reads these identical
    files. Note ``packed_row_fingerprint``'s internal read
    (``splice_counterfactual._file_fingerprint``) uses the builtin
    ``open()`` on a path, not ``Path.open`` -- it is not interceptable via
    this test's monkeypatch, so this schedule targets
    ``load_packed_row_shard``'s read specifically, the one packed-shard
    touch that is.

    B is structurally identical to A (same 1-segment layout, so the
    per-shard coupling check the loop performs still agrees with the
    selection shard's declared ``sel_offsets``) but has different token
    content, so it is the STAT-FENCE -- not a content-derived check -- that
    must catch this swap: a real, rename-based A->B->A schedule restores the
    exact original bytes (the recomputed ``packed_row_fingerprint`` and the
    coupling check both agree with the manifest and are content-only, so
    neither objects), but the restore lands a fresh inode, which the shard
    stat-fence's final recheck (after the last shard-body read) detects.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    (packed_shard_path,) = list(packed_dir.glob("*.safetensors.zst"))
    resolved = packed_shard_path.resolve()
    a_bytes = resolved.read_bytes()

    b_scratch_path = save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[9, 8, 7, 6, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 4], dtype=t.int32),
            feature_ids=t.tensor([1, 2, 4, 5], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180, 200, 90], dtype=t.uint8),
        ),
        tmp_path / ".staged_packed_b_source",
    )
    b_bytes = b_scratch_path.read_bytes()
    assert b_bytes != a_bytes
    b_source = tmp_path / ".staged_packed_shard_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_packed_shard_a_restore"
    a_restore_source.write_bytes(a_bytes)

    real_open = builtins.open
    swapped = {"done": False}

    def swapping_open(path: Any, *args: Any, **kwargs: Any):
        if (
            isinstance(path, (str, os.PathLike))
            and Path(path) == resolved
            and not swapped["done"]
        ):
            swapped["done"] = True
            os.replace(b_source, resolved)
            handle = real_open(path, *args, **kwargs)
            os.replace(a_restore_source, resolved)
            return handle
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match="changed",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )
    assert swapped["done"]
    assert resolved.read_bytes() == a_bytes


def test_selection_contract_rejects_a_packed_shard_segment_count_that_disagrees_with_the_selection_shard(
    tmp_path: Path,
) -> None:
    """Codex convergence-pass residual 1 (HIGH), packed-side reproduction,
    re-verified against the reverted production-parity implementation:
    Codex's own repro swapped the packed shard from A (1 segment, matching
    the selection shard's ``sel_offsets``) to B (2 segments) in the window
    between the per-shard semantic read (``load_packed_row_shard``, which
    sets ``expected_segments``) and the separate, later
    ``packed_row_fingerprint`` read -- accepted under the old code, because
    nothing bound those two reads to the same bytes.

    Here the packed shard is genuinely 2 segments from the START (no swap
    involved at all -- both reads see the identical, settled B state), so
    this proves the coupling check itself still correctly rejects a
    2-segment packed shard against a selection shard whose
    ``sel_offsets``/``neg_offsets`` declare only 1 segment, independent of
    whether shard bodies are read via a snapshot or by path -- routing both
    reads through live pathnames instead of a temp-path indirection did not
    scramble which shard's content feeds which check.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, _ = (
        _write_selection_fixture(tmp_path)
    )
    packed_dir = manifest_path.parent.parent.parent
    (packed_shard_path,) = list(packed_dir.glob("*.safetensors.zst"))
    # B: genuinely 2 segments (one row split in two), replacing A (1
    # segment) in place -- the selection shard's sel_offsets/neg_offsets
    # still declare exactly 1 segment (unchanged from the fixture), so the
    # canonical coupling check must now disagree.
    save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 2], dtype=t.int32),
            seg_token_start=t.tensor([0, 2], dtype=t.uint16),
            seg_token_end=t.tensor([2, 4], dtype=t.uint16),
            seg_doc_id=t.tensor([0, 0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 2, 4], dtype=t.int32),
            feature_ids=t.tensor([1, 2, 4, 5], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180, 200, 90], dtype=t.uint8),
        ),
        packed_shard_path,
    )
    _reissue_fixture_validation_certificate(
        manifest_path,
        clusters,
        segment_count=1,
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match=r"producer-certificate segment-count mismatch",
    ):
        _call_selection_contract(
            metadata,
            manifest_path,
            mask,
            clusters,
            consolidation,
            percentile_checksum,
            force_cluster_shard_revalidation=True,
        )


def test_selection_contract_rejects_a_symlinked_selection_shard(tmp_path: Path) -> None:
    """Codex pass-3 residual (MEDIUM), now closed by the production-parity
    revert: the removed temp-materialization resolved a shard's pathname
    before copying it (B1's ``stream_hash_file_stable`` calls
    ``Path.resolve(strict=True)``), silently laundering a symlink into a
    REGULAR temp file -- so a symlinked shard the canonical validator
    itself deliberately rejects
    (``feature_clusters._file_content_identity``'s ``stat.S_ISREG`` check,
    covered directly at that primitive's own level by
    ``test_clustered_selection_integrity_rejects_symlinked_shard`` in
    ``test_feature_clusters_identity_guards.py``) was silently accepted by
    ``_selection_contract`` anyway. With no materialization step, the
    canonical rejection fires directly against the live, still-symlinked
    path, exactly as it always has for
    ``dataloader._validate_generic_selection_dir`` -- this is the
    end-to-end regression proving that through ``_selection_contract``
    specifically, not just at the primitive's own unit level.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    # Outside-target symlink (Codex's exact pass-3 repro), with the target's
    # content identical to the original shard bytes -- isolates this test to
    # the symlink-ness alone, not a content/checksum mismatch.
    target = tmp_path / "outside_selection_dir_target.safetensors.zst"
    target.write_bytes(shard_path.read_bytes())
    shard_path.unlink()
    shard_path.symlink_to(target)
    with pytest.raises(ClusterFeatureContractError, match="must be a real regular file"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


# --------------------------------------------------------------------------
# ``_selection_contract`` — Codex convergence-pass residual 2 (MEDIUM):
# non-injective shard stem extraction (same NOT APPROVED review round as
# residual 1 above).
# --------------------------------------------------------------------------


def test_selection_contract_rejects_a_selection_directory_with_two_shards_that_normalize_to_the_same_stem(
    tmp_path: Path,
) -> None:
    """Codex convergence-pass residual 2 (MEDIUM), exact reproduction:
    ``path.stem.split(".")[0]`` collapsed a certified two-file selection
    inventory containing both ``<stem>.aaa.safetensors.zst`` and
    ``<stem>.safetensors.zst`` onto the same dict key, silently dropping one
    of the two from ``selection_by_stem`` (a plain dict comprehension keeps
    only the last glob-order path for a repeated key) -- so the
    packed/selection coverage check saw only ONE selection stem, matched it
    1:1 against the single packed shard, reported "exact coverage", and the
    per-shard loop never validated the dropped file at all.

    The corrected full-``COMPRESSED_EXT``-suffix strip in
    ``_unique_shard_stem_map`` distinguishes the two filenames
    (``shard_00000.aaa`` vs ``shard_00000``) instead of coalescing them, so
    the coverage check now correctly sees an untracked extra selection shard
    with no packed counterpart and rejects it outright -- no shard is
    silently skipped.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    assert shard_path.name == "shard_00000.safetensors.zst"
    colliding_path = shard_path.with_name("shard_00000.aaa.safetensors.zst")
    colliding_path.write_bytes(shard_path.read_bytes())
    with pytest.raises(ClusterFeatureContractError, match="must match packed shard stems exactly"):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_unique_shard_stem_map_rejects_two_paths_that_normalize_to_the_same_stem() -> None:
    """Direct unit coverage of ``_unique_shard_stem_map``'s explicit
    duplicate rejection (defense in depth, Codex convergence-pass residual 2
    MEDIUM): with the corrected full-suffix strip, two DISTINCT filenames
    from the same real directory listing can no longer collide
    (``str.removesuffix`` of one fixed literal is injective over distinct
    ``.name`` values), so this raise is unreachable through
    ``_selection_contract``'s own glob-based call sites (see
    ``test_selection_contract_rejects_a_selection_directory_with_two_shards_that_normalize_to_the_same_stem``
    above, which is instead caught by the pre-existing coverage check).
    Exercised directly here so the defensive branch is still proven correct
    rather than left untested -- matching this module's existing
    "dead-branch documentation" precedent (e.g.
    ``test_selection_contract_rejects_missing_shard_aggregate_identity``).
    """
    paths = [
        Path("/tmp/shards_a/x.safetensors.zst"),
        Path("/tmp/shards_b/x.safetensors.zst"),
    ]
    with pytest.raises(
        ClusterFeatureContractError,
        match="has two shards that normalize to the same stem 'x'",
    ):
        ccp._unique_shard_stem_map(paths, surface="test directory")


# --------------------------------------------------------------------------
# ``_selection_contract`` — zero-copy production-parity rewrite: torn/
# truncated shard coverage and the accepted same-inode threat-model
# boundary (see the module docstring's "Accepted threat-model boundary"
# paragraph).
# --------------------------------------------------------------------------


def test_selection_contract_rejects_a_truncated_selection_shard(tmp_path: Path) -> None:
    """Torn-write coverage for the production-parity revert: a genuinely
    truncated selection shard is rejected. The byte-integrity check's
    per-shard content hash/length no longer matches the manifest's declared
    (untouched) aggregate -- caught before the per-shard semantic loader or
    the stat-fence ever run, since ``assert_clustered_selection_shard_
    integrity`` is the first canonical read in the by-path sequence.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    original_bytes = shard_path.read_bytes()
    truncated = original_bytes[: len(original_bytes) // 2]
    assert 0 < len(truncated) < len(original_bytes)
    shard_path.write_bytes(truncated)
    with pytest.raises(ClusterFeatureContractError):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )


def test_selection_contract_accepts_a_stat_signature_spoofed_same_inode_shard_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documents the ACCEPTED scope boundary at production-dataloader parity
    (owner scope decision -- see the module docstring's "OWNER SCOPE
    DECISION" / "Accepted threat-model boundary" paragraphs), as a positive,
    TESTED assertion of the documented behavior rather than a silent gap.

    This is the exact same A->B->A schedule
    ``test_selection_contract_rejects_a_selection_shard_a_to_b_to_a_schedule_via_the_stat_fence``
    rejects, except the stat probe itself is defeated here:
    ``ccp._stat_signature`` is monkeypatched to report ONE fixed signature
    for this shard's path no matter what is actually on disk, at every call
    -- both the "before" capture and the "after" recheck. This simulates a
    same-inode write whose stat delta is not observable at the two specific
    instants this function samples it (not reproducible via real wall-clock
    timing on a real filesystem, so simulated directly rather than raced;
    see the module docstring for why closing this structurally would
    require fd-binding across three separate main-owned consumer functions,
    which is out of scope, and why the disproven temp-materialization
    approach is the only alternative way to simulate that binding).

    With the probe blinded, nothing left in this function can catch it: the
    byte-integrity check and the semantic validator each see only ONE of
    the two versions at their own read time (A -- the coordinatedly
    re-signed, consolidated-away-ID shard -- is what the manifest's own
    aggregate matches and what is on disk when the function returns; the
    semantic validator transiently saw valid B instead, only during the
    swap window around ``load_selection_shard``'s read), and the
    stat-fence -- the only mechanism spanning across those separate reads
    -- is exactly what is defeated here. The contract therefore returns
    SUCCESSFULLY, certifying a shard that (as of the moment it returns)
    contains a consolidated-away feature ID. This is the documented, TESTED
    floor of a detect-after-the-fact strategy at production-dataloader
    parity, not a silent gap: the eval-time dataloader this function
    mirrors has NO stat-fence at all, so it shares this exact residual
    unconditionally, on every real read -- not just this contrived,
    probe-defeated edge case.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    resolved = shard_path.resolve()
    b_bytes = resolved.read_bytes()  # the fixture's genuinely valid shard
    frozen_signature = ccp._stat_signature(resolved)

    # A: consolidated-away ID 5, coordinately re-signed so the manifest's
    # OWN byte-integrity aggregate matches these new bytes exactly -- same
    # setup as the stat-fence regression this test's docstring references.
    save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([1, 5], dtype=t.uint16),  # 5 is consolidated-away
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([3], dtype=t.uint16),
        ),
        shard_path,
    )
    _mutate_manifest(
        manifest_path,
        lambda m: m["metadata"].__setitem__(
            "selection_shards",
            build_clustered_selection_shard_integrity([shard_path]),
        ),
    )
    _reissue_fixture_validation_certificate(manifest_path, clusters)
    a_bytes = resolved.read_bytes()
    assert a_bytes != b_bytes
    b_source = tmp_path / ".staged_b_valid_spoofed"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore_spoofed"
    a_restore_source.write_bytes(a_bytes)

    real_stat_signature = ccp._stat_signature

    def spoofed_stat_signature(path: Path) -> tuple[int, int, int, int, int, int]:
        if path == resolved:
            return frozen_signature
        return real_stat_signature(path)

    monkeypatch.setattr(ccp, "_stat_signature", spoofed_stat_signature)

    real_open = Path.open
    open_count = {"n": 0}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            open_count["n"] += 1
            if open_count["n"] == 2:
                os.replace(b_source, resolved)
                handle = real_open(self, *args, **kwargs)
                os.replace(a_restore_source, resolved)
                return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    selection = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        percentile_checksum,
        force_cluster_shard_revalidation=True,
    )
    assert open_count["n"] >= 2
    # The documented gap: the contract returned successfully, but the shard
    # ACTUALLY on disk right now (A, restored) contains a consolidated-away
    # feature ID the semantic validator never saw, because it only ran
    # during the transient window when B was swapped in.
    assert resolved.read_bytes() == a_bytes
    assert selection.selection_shards_sha256 == str(
        build_clustered_selection_shard_integrity([shard_path])["aggregate_sha256"]
    )


def test_selection_contract_rejects_a_different_inode_shard_swap_staged_after_the_manifest_cold_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex pass-4 residual (HIGH): the shard stat-fence's final recheck
    (``_assert_shard_stat_signatures_unchanged``) used to run BEFORE the two
    manifest cold checks (``_assert_digest_stat_unchanged``) rather than
    after them -- so it was not actually this function's "final
    cold-check batch" / the genuine last operation before ``return`` the
    module docstring's "Accepted threat-model boundary" paragraph claimed.
    A shard swap staged in that now-uncovered trailing window -- after the
    fence's one recheck, anywhere through the two manifest checks and up to
    ``return`` itself -- went completely uncaught, even though it changes
    the shard's inode (the realistic case the fence is otherwise designed
    to always catch; see the sibling A->B->A stat-fence tests above).

    Codex's exact reproduction: both manifest cold checks complete
    normally, THEN ``os.replace()`` installs an invalid selection shard with
    a DIFFERENT inode, and ``_selection_contract`` returned successfully
    anyway -- only a separate, subsequent canonical check (outside this
    contract entirely) caught the swap. This is NOT the documented sub-tick
    same-inode residual
    (``test_selection_contract_accepts_a_stat_signature_spoofed_same_inode_shard_swap``
    above): the inode genuinely changes here (a real ``os.replace`` rename,
    no stat-probe spoofing at all) and there is no race -- it
    deterministically escaped because nothing re-``lstat``ed the shards
    again after the fence's one (too-early) recheck.

    Hooks ``_assert_digest_stat_unchanged`` (not ``Path.open`` like the
    sibling stat-fence tests above): unlike those, this swap is staged
    entirely AFTER every shard-body read this function performs has already
    completed, so there is no shard read left to intercept by then -- only
    the two manifest cold checks and the shard fence's own final recheck
    remain. The real check runs first each time (the manifest is never
    actually tampered), and the swap fires only once BOTH calls have
    completed -- "both manifest checks pass, then the swap" -- matching
    Codex's observed schedule exactly rather than a timing race.

    Fix: the stat-fence recheck is now the genuine final operation before
    ``return`` (moved after both manifest cold checks), so this schedule --
    swap staged strictly AFTER both manifest checks and BEFORE the fence
    loop's first iteration -- is now caught by the fence's own inode-change
    detection. This is a real, permanent closure of the exact schedule
    Codex reproduced in pass-4, but it is not a closure of every trailing
    window: Codex pass-5 subsequently found two further schedules INSIDE
    the fence loop's own execution that this reorder does not cover -- see
    ``test_selection_contract_accepts_a_sequential_shard_fence_swap_of_an_already_rechecked_shard``
    and
    ``test_selection_contract_accepts_a_selection_manifest_swap_staged_during_the_shard_fence_loop``
    below, and the module docstring's "Accepted threat-model boundary"
    section for why no reordering closes those too.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    resolved = shard_path.resolve()
    valid_bytes = resolved.read_bytes()
    before_ino = resolved.stat().st_ino

    # Content is irrelevant here -- by the time this fires, every
    # content-based check this function performs (byte-integrity, the
    # per-shard semantic validator, the packed-row fingerprint) has already
    # run against the genuine, untouched shard. Only the DIFFERENT INODE
    # `os.replace` installs (a real rename, never the documented same-inode
    # residual) needs to be realistic; truncating also marks the swapped-in
    # file unambiguously "invalid", matching Codex's own framing of what was
    # installed.
    invalid_source = tmp_path / ".staged_invalid_different_inode"
    invalid_source.write_bytes(valid_bytes[: len(valid_bytes) // 2])

    real_assert_digest_stat_unchanged = ccp._assert_digest_stat_unchanged
    call_count = {"n": 0}

    def swapping_assert_digest_stat_unchanged(digest: cci.StableFileDigest) -> None:
        real_assert_digest_stat_unchanged(digest)
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Codex's exact schedule: BOTH manifest cold checks (selection,
            # then packed) have now completed normally; the swap happens
            # only afterward, in the window this fix must now cover.
            os.replace(invalid_source, resolved)

    monkeypatch.setattr(ccp, "_assert_digest_stat_unchanged", swapping_assert_digest_stat_unchanged)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(
            "^a selection/packed shard changed while this contract was "
            f"validating it: {re.escape(str(resolved))}$"
        ),
    ):
        _call_selection_contract(
            metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
        )
    assert call_count["n"] == 2
    assert resolved.stat().st_ino != before_ino


# --------------------------------------------------------------------------
# ``_selection_contract`` — Codex pass-5 (fifth review round, verifying the
# pass-4 reorder immediately above): two further trailing-window residuals,
# both HIGH, both INSIDE the final recheck batch pass-4 had just finished
# ordering. Preserved as permanent DOCUMENTED-ACCEPTED-BOUNDARY tests (like
# the sub-tick same-inode test above), not further "fixed" by reordering
# again — see the module docstring's "Codex pass-5 finding" / "Accepted
# threat-model boundary" sections for why no ordering of N>1 sequential
# by-path checks closes either window, and why dataloader-parity resolves
# both by documentation instead.
# --------------------------------------------------------------------------


def test_selection_contract_accepts_a_sequential_shard_fence_swap_of_an_already_rechecked_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex pass-5 finding 1 (HIGH). DOCUMENTED-ACCEPTED-BOUNDARY, at
    production-dataloader parity (owner scope decision — see the module
    docstring's "OWNER SCOPE DECISION" / "Accepted threat-model boundary"
    (point (c)) paragraphs), as a positive, TESTED assertion of the
    documented behavior rather than a silent gap.

    ``_assert_shard_stat_signatures_unchanged`` rechecks every shard in ONE
    sequential ``for`` loop over ``all_shard_paths`` (selection shard(s)
    first, packed shard(s) second). This loop is not atomic: once the
    selection shard's own iteration has passed, nothing loops back to
    recheck it again while a LATER iteration (the packed shard, here) is
    still running.

    Codex's exact schedule: hook ``_stat_signature`` keyed to the packed
    shard's path. Its FIRST call is ``_shard_stat_signatures``'s early
    "before" capture; its SECOND is the final return fence's packed-shard
    recheck — by which
    point this SAME loop has already rechecked and cleared the selection
    shard, earlier in iteration order. On that second call,
    replace the selection shard — already cleared — with a real,
    different-inode, truncated file, before returning the packed shard's
    own (unaffected, genuine) signature. ``_selection_contract`` returns
    successfully anyway: nothing re-``lstat``s the selection shard a second
    time once the loop has moved on to the packed shard.

    This is NOT the pass-4 schedule
    (``test_selection_contract_rejects_a_different_inode_shard_swap_staged_after_the_manifest_cold_checks``
    above): that swap was staged strictly BEFORE the fence loop's first
    iteration (right after both manifest checks completed, before either
    shard was rechecked at all). This one is staged INSIDE the loop's own
    execution, between two of its iterations — a window the pass-4 reorder
    was never able to cover, because no ordering of a sequential per-item
    loop can protect an earlier item from a swap timed to a later item's own
    check (see the module docstring for the full argument).

    Accepted resolution: dataloader parity, not further reordering. The
    swapped-in content is proven below to fail the SAME canonical integrity
    check (``assert_clustered_selection_shard_integrity``) that
    ``dataloader._validate_generic_selection_dir`` calls against this exact
    file — the swap is caught one call later than this preflight, not
    silently unenforced end-to-end.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    selection_path = shard_path.resolve()
    packed_dir = manifest_path.parent.parent.parent
    (packed_shard_path,) = list(packed_dir.glob("*.safetensors.zst"))
    packed_path = packed_shard_path.resolve()

    valid_bytes = selection_path.read_bytes()
    before_ino = selection_path.stat().st_ino
    # The manifest's OWN declared aggregate (what the returned contract will
    # keep echoing back) captured NOW, from the genuine bytes, before any
    # swap — this is what "the contract still reports the pre-swap identity"
    # means below, not a claim about what re-hashing the live file today
    # would produce (which the backstop check, not this value, proves).
    expected_aggregate = str(
        build_clustered_selection_shard_integrity([shard_path])["aggregate_sha256"]
    )
    invalid_bytes = valid_bytes[: len(valid_bytes) // 2]
    assert 0 < len(invalid_bytes) < len(valid_bytes)
    invalid_source = tmp_path / ".staged_sequential_escape_invalid_selection"
    invalid_source.write_bytes(invalid_bytes)

    real_stat_signature = ccp._stat_signature
    packed_calls = {"n": 0}
    swapped = {"done": False}

    def staged_stat_signature(path: Path) -> tuple[int, int, int, int, int, int]:
        if path == packed_path:
            packed_calls["n"] += 1
            if packed_calls["n"] == 2:
                # Exact schedule in the final return fence: the loop has already
                # rechecked (and cleared) the selection shard -- earlier in
                # iteration order -- and moved on to the packed shard's own
                # final recheck. Swap the selection shard NOW, while nothing
                # in this function is looking at it anymore.
                os.replace(invalid_source, selection_path)
                swapped["done"] = True
        return real_stat_signature(path)

    monkeypatch.setattr(ccp, "_stat_signature", staged_stat_signature)

    selection = _call_selection_contract(
        metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
    )

    # Documented-accepted behavior: the contract returns successfully even
    # though the selection shard actually live on disk right now is the
    # truncated, different-inode invalid file -- not what was validated.
    assert swapped["done"]
    assert packed_calls["n"] == 2
    assert selection_path.stat().st_ino != before_ino
    assert selection_path.read_bytes() == invalid_bytes
    assert selection.selection_shards_sha256 == expected_aggregate

    # Dataloader-parity backstop: an immediate re-check via the SAME
    # canonical integrity validator `dataloader._validate_generic_
    # selection_dir` calls against this exact file (confirmed by direct
    # read of that function) rejects the swapped-in content -- proving the
    # swap is caught one call later than this preflight, not silently
    # unenforced end-to-end.
    manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]
    with pytest.raises(FeatureClustersError):
        assert_clustered_selection_shard_integrity(
            manifest_path.parent,
            manifest_metadata,
            surface="dataloader-parity backstop re-check",
        )


def test_selection_contract_accepts_a_selection_manifest_swap_staged_during_the_shard_fence_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex pass-5 finding 2 (HIGH). DOCUMENTED-ACCEPTED-BOUNDARY, at
    production-dataloader parity (owner scope decision — see the module
    docstring's "OWNER SCOPE DECISION" / "Accepted threat-model boundary"
    (point (c)) paragraphs), as a positive, TESTED assertion of the
    documented behavior rather than a silent gap.

    The final return fence runs both manifest cold-checks
    (``_assert_digest_stat_unchanged``) BEFORE the shard stat-fence's loop,
    so within that fence neither manifest is rechecked again
    AFTER that loop starts. This is the mirror image of combined-review finding 3
    (the original gap the manifest cold-check batch was added to close):
    closing the pass-4 shard-swap schedule reopened this one, because it
    made the shard fence's loop — not the manifest checks — the thing that
    now runs last and takes nonzero time.

    Codex's exact schedule: hook ``_stat_signature`` keyed to the SELECTION
    SHARD's path, not the manifest — manifests are only stat-checked via
    B1's ``_assert_digest_stat_unchanged``, which resolves its internal
    ``_stat_signature`` call against ``cluster_contract_identity.py``'s own
    module namespace (where that function is defined), not this module's
    ``ccp._stat_signature`` — so patching the latter here leaves both
    manifest cold-checks running for real, unaffected, exactly like Codex's
    own reproduction. The selection shard's FIRST ``_stat_signature`` call
    is ``_shard_stat_signatures``'s early "before" capture, long before
    either manifest check runs; its SECOND call is the final fence's
    shard-loop first iteration
    (selection shard(s) come first in ``all_shard_paths``), after that final
    fence's manifest cold-checks completed normally. On that second call,
    replace the selection MANIFEST — already cleared
    by its own cold-check — with a real, different-inode, truncated file.
    ``_selection_contract`` returns successfully anyway, and the returned
    ``selection_manifest_file_sha256`` describes bytes no longer live on
    disk.

    Accepted resolution: dataloader parity, not further reordering. The
    swapped-in content is proven below to fail an immediate reload the same
    way ``dataloader._validate_generic_selection_dir``'s own
    ``read_manifest(selection_dir)`` call would (returns ``None`` on invalid
    JSON, which the dataloader's own null-check turns into a load-time
    ``ValueError``) — the swap is caught one call later than this
    preflight, not silently unenforced end-to-end.
    """
    metadata, manifest_path, mask, clusters, consolidation, percentile_checksum, shard_path = (
        _write_selection_fixture(tmp_path)
    )
    manifest_path = manifest_path.resolve()
    selection_path = shard_path.resolve()

    valid_manifest_bytes = manifest_path.read_bytes()
    before_ino = manifest_path.stat().st_ino
    invalid_manifest_bytes = valid_manifest_bytes[: len(valid_manifest_bytes) // 2]
    assert 0 < len(invalid_manifest_bytes) < len(valid_manifest_bytes)
    invalid_manifest_source = tmp_path / ".staged_manifest_finality_invalid"
    invalid_manifest_source.write_bytes(invalid_manifest_bytes)

    real_stat_signature = ccp._stat_signature
    selection_calls = {"n": 0}
    swapped = {"done": False}

    def staged_stat_signature(path: Path) -> tuple[int, int, int, int, int, int]:
        if path == selection_path:
            selection_calls["n"] += 1
            if selection_calls["n"] == 2:
                # Codex's exact schedule: both manifest cold-checks have
                # already completed normally (current pass-4 ordering runs
                # them before the shard fence loop), and the shard fence
                # loop's first iteration (the selection shard) is now
                # rechecking. Swap the manifest NOW, while nothing in this
                # function is looking at it anymore.
                os.replace(invalid_manifest_source, manifest_path)
                swapped["done"] = True
        return real_stat_signature(path)

    monkeypatch.setattr(ccp, "_stat_signature", staged_stat_signature)

    selection = _call_selection_contract(
        metadata, manifest_path, mask, clusters, consolidation, percentile_checksum
    )

    # Documented-accepted behavior: the contract returns successfully even
    # though the selection manifest actually live on disk right now is the
    # truncated, different-inode invalid file -- not what
    # `selection_manifest_file_sha256` describes.
    assert swapped["done"]
    assert selection_calls["n"] == 2
    assert manifest_path.stat().st_ino != before_ino
    assert manifest_path.read_bytes() == invalid_manifest_bytes
    assert (
        selection.selection_manifest_file_sha256 == hashlib.sha256(valid_manifest_bytes).hexdigest()
    )
    assert (
        selection.selection_manifest_file_sha256
        != hashlib.sha256(invalid_manifest_bytes).hexdigest()
    )

    # Dataloader-parity backstop, two independent proofs that the swap is
    # caught one call later than this preflight, not silently unenforced
    # end-to-end:
    # (1) the literal function `dataloader._validate_generic_selection_dir`
    #     calls first (`read_manifest(selection_dir)`) returns None on this
    #     exact invalid JSON, which the dataloader's own null-check turns
    #     into a load-time ValueError.
    assert read_manifest(manifest_path.parent) is None
    # (2) this module's own equivalent manifest-loading primitive (the same
    #     fused hash-once/read-once/content-verified check
    #     `_selection_contract` itself used to load this exact manifest)
    #     independently rejects the swapped-in content just as loudly.
    with pytest.raises(ClusterFeatureContractError, match="could not read"):
        ccp._load_json_mapping(manifest_path, surface="dataloader-parity backstop re-check")


def _files_pin(model_path: Path, metadata_path: Path) -> ccp.CheckpointFilesPin:
    return ccp.CheckpointFilesPin(
        checkpoint_dir=model_path.parent,
        resolved_tag=model_path.parent.name,
        step=0,
        model_file=cci.stream_hash_file_stable(model_path),
        metadata_file=cci.stream_hash_file_stable(metadata_path),
    )


def test_pin_recheck_rejects_swap_that_restores_content_and_mtime(tmp_path: Path) -> None:
    """``st_ctime_ns`` is the load-bearing field, so pin it directly.

    ``assert_checkpoint_files_pin_unchanged`` compares the whole six-field
    ``lstat`` signature, but only one of those fields actually survives a
    determined A->B->A round trip, and the sibling regression in
    ``tests/axbench_base/unit/test_runtime_state.py`` happens to be caught by
    ``st_ino``. That leaves the real guarantee untested: drop ``st_ctime_ns``
    from the comparison and that sibling still passes.

    So this test constructs the strongest round trip a non-privileged writer
    can perform -- ``shutil.copy2`` preserves mtime across the restoring rename
    -- and asserts up front that content, byte length, size and mtime all match
    again afterwards. Every weaker field is therefore satisfied by the time the
    recheck runs, and only ctime can reject it. ctime is kernel-maintained on
    both content writes and renames into a path, and POSIX exposes no setter
    for it (``os.utime`` moves atime/mtime, and bumps ctime doing so).
    """
    model_path = tmp_path / "model_000000.pt"
    metadata_path = tmp_path / "meta_000000.json"
    model_path.write_bytes(b"AAAA-original-weights")
    metadata_path.write_text("{}", encoding="utf-8")

    # copy2 carries mtime, so the restore below cannot be caught by mtime.
    pristine = tmp_path / "pristine.keep"
    shutil.copy2(model_path, pristine)
    pin = _files_pin(model_path, metadata_path)

    # A -> B, then B -> A with mtime carried back across the rename.
    replacement = tmp_path / "b.tmp"
    replacement.write_bytes(b"BBBB-different-weights")
    replacement.replace(model_path)
    restore = tmp_path / "a.tmp"
    shutil.copy2(pristine, restore)
    restore.replace(model_path)

    # Guard against a degenerate fixture: if any weaker field already differs,
    # this test would pass without exercising ctime at all.
    after = cci.stream_hash_file_stable(model_path)
    assert after.sha256 == pin.model_file.sha256
    assert after.byte_length == pin.model_file.byte_length
    mode, dev, _ino, size, mtime_ns, ctime_ns = after.stat_signature
    p_mode, p_dev, _p_ino, p_size, p_mtime_ns, p_ctime_ns = pin.model_file.stat_signature
    assert (mode, dev, size, mtime_ns) == (p_mode, p_dev, p_size, p_mtime_ns)
    assert ctime_ns != p_ctime_ns, "ctime must be the sole surviving discriminator"

    with pytest.raises(
        ClusterFeatureContractError,
        match="checkpoint model bytes changed after they were pinned",
    ):
        ccp.assert_checkpoint_files_pin_unchanged(pin)


def test_pin_recheck_accepts_untouched_files(tmp_path: Path) -> None:
    """False-positive control for the stat-signature tightening.

    A read-only load window must not trip the recheck; without this, the test
    above is satisfied by a recheck that rejects everything.
    """
    model_path = tmp_path / "model_000000.pt"
    metadata_path = tmp_path / "meta_000000.json"
    model_path.write_bytes(b"AAAA-original-weights")
    metadata_path.write_text("{}", encoding="utf-8")
    pin = _files_pin(model_path, metadata_path)

    assert model_path.read_bytes() == b"AAAA-original-weights"
    ccp.assert_checkpoint_files_pin_unchanged(pin)


# --------------------------------------------------------------------------
# Splice-counterfactual support in the clustered selection preflight.
#
# Splice-counterfactual training with neg_delta_ratio=1.0 produces the same
# conditioning semantics as an undelta'd run for the feature-space methods
# clustered evaluation drives (encoder-decoder DiffMean and GCG), so its
# checkpoints are evaluable. Every other mode, and any partial ratio, stays
# rejected.
# --------------------------------------------------------------------------


_SUPPORTED_DELTA = {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}


def _splice_user_config(**overrides: Any) -> dict[str, Any]:
    user_config = _selection_user_config(feature_delta=dict(_SUPPORTED_DELTA))
    user_config["feature_selection"] = {
        "distribution": "geometric",
        "mean": 2.0,
        "delta_weights": "percentile_delta",
    }
    user_config.update(overrides)
    return user_config


def test_clustered_selection_accepts_splice_counterfactual_with_delta_weights(
    tmp_path: Path,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    identity = ccp._resolve_selection_identity(
        metadata={"user_config": _splice_user_config()},
        feature_mask=mask,
        feature_clusters=clusters,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
    )
    assert identity.selection_hash


def test_clustered_selection_accepts_splice_counterfactual_default_ratio(
    tmp_path: Path,
) -> None:
    """neg_delta_ratio defaults to 1.0, so an omitted key is still supported."""

    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config(feature_delta={"mode": "splice_counterfactual"})
    identity = ccp._resolve_selection_identity(
        metadata={"user_config": user_config},
        feature_mask=mask,
        feature_clusters=clusters,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
    )
    assert identity.selection_hash


@pytest.mark.parametrize(
    ("feature_delta", "match"),
    [
        # Supported mode, partial ratio -- different conditioning semantics.
        (
            {"mode": "splice_counterfactual", "neg_delta_ratio": 0.5},
            "supports only feature_delta.mode",
        ),
        ({"mode": "prev_segment"}, "supports only feature_delta.mode"),
        ({"mode": "cum_max"}, "supports only feature_delta.mode"),
        # Malformed blocks surface as a wrapped parse error, not as silent
        # acceptance of an unvalidated dict.
        ({"enabled": True}, "invalid checkpoint feature_delta"),
        ({"mode": "not_a_mode"}, "invalid checkpoint feature_delta"),
        (
            {"mode": "splice_counterfactual", "neg_delta_ratio": 2.0},
            "invalid checkpoint feature_delta",
        ),
    ],
)
def test_clustered_selection_still_rejects_unsupported_feature_delta(
    tmp_path: Path,
    feature_delta: dict[str, Any],
    match: str,
) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config(feature_delta=feature_delta)
    with pytest.raises(ClusterFeatureContractError, match=match):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_clustered_selection_rejects_a_non_mapping_feature_delta(tmp_path: Path) -> None:
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config(feature_delta="splice_counterfactual")
    with pytest.raises(ClusterFeatureContractError, match="must be a mapping or null"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_clustered_selection_still_rejects_delta_weights_without_a_feature_delta(
    tmp_path: Path,
) -> None:
    """delta_weights alone is unverifiable -- there is no delta block to check."""

    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config(feature_delta=None)
    with pytest.raises(ClusterFeatureContractError, match="delta_weights disabled"):
        ccp._resolve_selection_identity(
            metadata={"user_config": user_config},
            feature_mask=mask,
            feature_clusters=clusters,
            gcg_top_k_pos=None,
            gcg_top_k_neg=None,
        )


def test_splice_selection_hash_matches_the_producer_not_the_null_delta_form(
    tmp_path: Path,
) -> None:
    """The recomputed hash must be the one the packed-row producer wrote.

    `delta_config` is an unconditional payload key of `compute_selection_hash`,
    so hashing `None` for a splice-trained checkpoint yields a hash no producer
    ever wrote -- `resolve_selection_manifest_path` would then find zero
    `sel/<hash>/manifest.json` candidates. This pins the site against the
    producer call in `scripts/preprocess/precompute_selected_features.py`, and
    pins that the two forms genuinely differ (otherwise the first assert would
    pass vacuously).
    """

    from steering.packed_batch_io import compute_selection_hash

    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config()
    identity = ccp._resolve_selection_identity(
        metadata={"user_config": user_config},
        feature_mask=mask,
        feature_clusters=clusters,
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
    )
    producer_kwargs: dict[str, Any] = {
        "positive_selection": user_config["feature_selection"],
        "negative_selection": user_config["negative_selection"],
        "num_buckets": 2,
        "seed": 42,
        "ddp_rank": 0,
        "feature_mask": canonical_feature_mask_dict(mask),
        "feature_clusters": canonical_feature_clusters_descriptor(clusters),
        "splice_batch_size": None,
    }
    producer_hash = compute_selection_hash(delta_config=dict(_SUPPORTED_DELTA), **producer_kwargs)
    null_delta_hash = compute_selection_hash(delta_config=None, **producer_kwargs)

    assert identity.selection_hash == producer_hash
    assert producer_hash != null_delta_hash


def test_splice_selection_hash_honours_the_producer_batch_size(tmp_path: Path) -> None:
    """`selection_splice_batch_size` must reach the hash, or discovery misses.

    The producer's batch size participates in the selection hash but is not
    recorded in checkpoint metadata, so it can only come from the caller. A
    batched producer therefore needs this override; without it the recomputed
    hash is the unbatched one and `resolve_selection_manifest_path` finds zero
    candidates.
    """

    from steering.packed_batch_io import compute_selection_hash

    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    clusters = _clusters(tmp_path / "clusters.yaml")
    user_config = _splice_user_config()
    identity_kwargs: dict[str, Any] = {
        "metadata": {"user_config": user_config},
        "feature_mask": mask,
        "feature_clusters": clusters,
        "gcg_top_k_pos": None,
        "gcg_top_k_neg": None,
    }
    unbatched = ccp._resolve_selection_identity(**identity_kwargs)
    batched = ccp._resolve_selection_identity(**identity_kwargs, selection_splice_batch_size=64)
    assert batched.selection_hash != unbatched.selection_hash

    producer_kwargs: dict[str, Any] = {
        "positive_selection": user_config["feature_selection"],
        "negative_selection": user_config["negative_selection"],
        "delta_config": dict(_SUPPORTED_DELTA),
        "num_buckets": 2,
        "seed": 42,
        "ddp_rank": 0,
        "feature_mask": canonical_feature_mask_dict(mask),
        "feature_clusters": canonical_feature_clusters_descriptor(clusters),
    }
    assert batched.selection_hash == compute_selection_hash(splice_batch_size=64, **producer_kwargs)
    assert unbatched.selection_hash == compute_selection_hash(
        splice_batch_size=None, **producer_kwargs
    )


def test_resolve_and_validate_sae_accepts_an_org_qualified_loaded_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gemma-scope-2 SAEs report a fully-qualified HF id.

    The checkpoint records the bare feature-path token (the form
    ``sanitize_model_name`` round-trips through the ``m_*`` directory
    component), so an exact string comparison rejects a correctly-matched
    pair. GemmaScope v1 reports the bare name on both sides, which is why this
    held until gemma-scope-2.
    """

    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            model_name="google/gemma-2-2b",  # org-qualified; checkpoint says gemma-2-2b
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    identity = ccp._resolve_and_validate_sae(
        metadata=_sae_metadata(),
        feature_directory=_sae_feature_directory(),
        n_sae_features=16384,
        sae_loader=lambda _resolved: sae,
    )
    # The checkpoint's bare token stays the recorded identity -- accepting the
    # qualified form must not rewrite what the contract pins.
    assert identity.model == "gemma-2-2b"


def test_resolve_and_validate_sae_still_rejects_a_qualified_but_different_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring the ``org/`` prefix must not weaken the model check itself."""

    monkeypatch.setattr(ccp, "resolve_sae", lambda **_kwargs: _SAE_RESOLVED)
    sae = SimpleNamespace(
        W_dec=t.zeros((16384, 2)),
        cfg=SimpleNamespace(
            d_in=2,
            model_name="google/gemma-2-9b",  # WRONG model, merely org-qualified
            metadata=SimpleNamespace(hook_name="blocks.8.hook_resid_post"),
        ),
    )
    with pytest.raises(ClusterFeatureContractError, match="loaded SAE model_name"):
        ccp._resolve_and_validate_sae(
            metadata=_sae_metadata(),
            feature_directory=_sae_feature_directory(),
            n_sae_features=16384,
            sae_loader=lambda _resolved: sae,
        )


# --------------------------------------------------------------------------
# Auxiliary feature-prediction head.
#
# `feature_pred_head.*` is an auxiliary training-loss output, not part of the
# conditioning path. The fair and roofline encoder-decoder DiffMean / GCG
# methods never read it, so a checkpoint carrying one is evaluable under
# exactly those methods.
# --------------------------------------------------------------------------


_FP_HEAD = {"block": "mlp", "head_layers": 0, "weight": 0.1, "layer": -1, "loss_mode": "max_pool", "loss_fn": "bce"}


def _fp_identity(user_config: dict[str, Any], tmp_path: Path, **kwargs: Any) -> Any:
    return ccp._resolve_selection_identity(
        metadata={"user_config": user_config},
        feature_mask=FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6)),
        feature_clusters=_clusters(tmp_path / "clusters.yaml"),
        gcg_top_k_pos=None,
        gcg_top_k_neg=None,
        **kwargs,
    )


def test_feature_prediction_accepted_for_the_fair_and_roofline_methods(
    tmp_path: Path,
) -> None:
    user_config = _selection_user_config(feature_prediction=dict(_FP_HEAD))
    identity = _fp_identity(
        user_config,
        tmp_path,
        enabled_methods=CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS,
    )
    assert identity.selection_hash


def test_feature_prediction_head_does_not_change_the_selection_hash(
    tmp_path: Path,
) -> None:
    """The head is not a selection input, so admitting it must not move the hash.

    This is what makes the narrowing safe for artifact discovery: unlike the
    delta block, `feature_prediction` never reaches `compute_selection_hash`.
    """

    with_head = _fp_identity(
        _selection_user_config(feature_prediction=dict(_FP_HEAD)),
        tmp_path,
        enabled_methods=CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS,
    )
    without_head = _fp_identity(_selection_user_config(), tmp_path)
    assert with_head.selection_hash == without_head.selection_hash


def test_feature_prediction_rejected_when_methods_are_not_declared(
    tmp_path: Path,
) -> None:
    """An unstated method set is not assumed safe."""

    user_config = _selection_user_config(feature_prediction=dict(_FP_HEAD))
    with pytest.raises(ClusterFeatureContractError, match="declare enabled_methods"):
        _fp_identity(user_config, tmp_path)


@pytest.mark.parametrize(
    "enabled_methods",
    [
        # An encdec method outside the head-agnostic set, mixed in with safe
        # ones, is still rejected.
        (*CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS, "ours_replace_all_encdec"),
        ("ours_extract_and_add_encdec",),
        ("control_encdec",),
    ],
)
def test_feature_prediction_rejected_for_methods_outside_the_safe_set(
    tmp_path: Path,
    enabled_methods: tuple[str, ...],
) -> None:
    user_config = _selection_user_config(feature_prediction=dict(_FP_HEAD))
    with pytest.raises(ClusterFeatureContractError, match="additionally requests"):
        _fp_identity(user_config, tmp_path, enabled_methods=enabled_methods)


def test_enabled_methods_is_inert_without_a_feature_prediction_head(
    tmp_path: Path,
) -> None:
    """A head-free checkpoint is unaffected by the declared method set."""

    baseline = _fp_identity(_selection_user_config(), tmp_path)
    declared = _fp_identity(
        _selection_user_config(),
        tmp_path,
        enabled_methods=("ours_extract_and_add_encdec",),
    )
    assert declared.selection_hash == baseline.selection_hash


def test_feature_prediction_accepted_for_decoder_only_methods(tmp_path: Path) -> None:
    """Decoder-only methods are out of scope for the head by construction.

    The head lives on the encoder-decoder checkpoint this contract pins. The
    decoder-only methods run against a separate model that has no such head,
    so they cannot reach it -- this is not an allowlist judgement about whether
    they happen to read it.
    """

    user_config = _selection_user_config(feature_prediction=dict(_FP_HEAD))
    identity = _fp_identity(
        user_config,
        tmp_path,
        enabled_methods=CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
    )
    assert identity.selection_hash


def test_feature_prediction_accepted_for_mixed_decoder_only_and_safe_encdec(
    tmp_path: Path,
) -> None:
    """A clustered source run plus head-agnostic encdec methods is fine."""

    user_config = _selection_user_config(feature_prediction=dict(_FP_HEAD))
    identity = _fp_identity(
        user_config,
        tmp_path,
        enabled_methods=(
            *CLUSTERED_AXBENCH_BASE_SOURCE_METHODS,
            *CLUSTERED_FEATURE_PREDICTION_SAFE_METHODS,
        ),
    )
    assert identity.selection_hash


def test_clustered_encdec_target_methods_agrees_with_the_method_registry() -> None:
    """Drift sentinel for the derived encdec-target constant.

    `CLUSTERED_ENCDEC_TARGET_METHODS` is derived in `cluster_config` so that
    module stays free of a dependency on the AxBench-Base registry. The
    registry's `Method.target_model` is the authority, so the two must agree
    for every clustered method the registry knows -- in BOTH directions, so a
    method silently changing target is caught.
    """

    from steering.eval.axbench_base.method_registry import ALL_METHODS

    target_by_name = {method.name: method.target_model for method in ALL_METHODS}

    for name in CLUSTERED_ENCDEC_TARGET_METHODS:
        if name in target_by_name:
            assert target_by_name[name] == "encdec", (
                f"{name} is in CLUSTERED_ENCDEC_TARGET_METHODS but the registry "
                f"says target_model={target_by_name[name]!r}"
            )
    for name in CLUSTERED_AXBENCH_BASE_SOURCE_METHODS:
        assert name not in CLUSTERED_ENCDEC_TARGET_METHODS
        if name in target_by_name:
            assert target_by_name[name] == "decoder_only", (
                f"{name} is treated as decoder-only but the registry says "
                f"target_model={target_by_name[name]!r}"
            )


def test_selection_contract_retains_checkpoint_delta_for_replay(
    tmp_path: Path,
) -> None:
    """A delta-weighted checkpoint's delta config must survive into the contract.

    Clustered context-cache reconstruction rebuilds the checkpoint's feature
    pipeline from this contract. ``feature_pipeline`` cross-validates the pair,
    so if ``delta_weights`` is recorded but the delta config is dropped, replay
    fails with "SamplingSelection has delta_weights set but feature_delta is
    None" -- which is exactly what the real l22 clustered acceptance hit.

    Reuses ``legacy_splice_certificate`` rather than adding a parameter: it
    already produces the delta config this needs, so a second knob would be
    redundant surface.
    """
    metadata, manifest_path, mask, clusters, consolidation, checksum, _ = (
        _write_selection_fixture(
            tmp_path,
            positive={
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "delta_weights": "percentile_delta",
            },
            legacy_splice_certificate=True,
        )
    )

    selection = _call_selection_contract(
        metadata,
        manifest_path,
        mask,
        clusters,
        consolidation,
        checksum,
    )

    expected = {"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
    assert selection.feature_delta_json is not None, (
        "the contract must retain the checkpoint's delta config, or clustered "
        "replay of a delta-weighted checkpoint cannot reconstruct its pipeline"
    )
    assert json.loads(selection.feature_delta_json) == expected
    # canonical_dict stays null by design: identity must not shift for a field
    # that adds no distinguishing power over positive_selection's delta_weights.
    assert selection.canonical_dict()["feature_delta"] is None
    # And the typed accessor must hand back a validated object, not raw JSON.
    parsed = selection.parsed_feature_delta()
    assert parsed is not None
    # A validated object with an enum mode, not a dict of strings -- which is
    # the point of the accessor: the JSON stays canonical for hashing while
    # callers get the real type.
    assert parsed.mode is DeltaMode.splice_counterfactual
    assert parsed.neg_delta_ratio == 1.0
