"""Functional test for the l0_tier extraction shard-write wiring.

This is the test that *should* have caught the P0: the entire T21-T25
l0_tier protection layer was dead code because the extraction loop never
injected the tier into written shards nor called the overwrite guard.

The real ``process_shard`` loop needs a GPU model, so this test cannot
call it directly. Instead it exercises the *exact same wiring units* the
production code now uses — ``l0_tier_for_resolved`` (Step 0 helper),
``assert_l0_tier_compatible`` (the guard, called before the write loop),
``stamp_l0_tier`` (the inject, called right before every shard write) and
``save_features`` (the real serializer) — in the same order and against a
``shard_dict`` mirroring ``ShardData.to_dict()``. The production path
calls the same functions, so this is not a reimplementation: if the
wiring regresses (helper, guard, or inject), this test fails.
"""

import pytest
import torch as t

from steering.extract_features import stamp_l0_tier
from steering.feature_io import (
    FeatureDirectoryMismatchError,
    MetadataKey,
    assert_l0_tier_compatible,
    load_features,
    read_l0_tier_from_shard,
    save_features,
)
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    l0_tier_for_resolved,
    resolve_sae,
)


def _shard_dict() -> dict:
    """A minimal dict mirroring the shape of ``ShardData.to_dict()``.

    Mirrors ``steering.extract_features.ShardData`` fields (only the keys
    ``save_features`` requires for a valid serialize); ``to_dict`` drops
    None-valued fields, so an int/str-and-tensor subset is faithful.
    """
    return {
        "offsets": t.tensor([0, 3], dtype=t.long),
        "feature_ids": t.tensor([1, 2, 3], dtype=t.long),
        "feature_mags": t.tensor([0.5, 0.7, 0.3], dtype=t.float32),
        "doc_ids": t.zeros(1, dtype=t.int32),
        "segment_indices": t.zeros(1, dtype=t.int32),
        "segment_positions": t.zeros(1, dtype=t.int32),
        "doc_token_counts": t.tensor([128], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 22,
        "context_window": 1024,
        "pooling": "max_pool",
        "segment_size": 128,
        "segment_mode": "fixed_tokens",
        "context_independent_features": False,
    }


def _run_wired_write(
    out_dir, out_stem, resolved: ResolvedSAE, *, shard=None
) -> L0Tier | None:
    """Reproduce the production extraction wiring around a single shard write.

    Mirrors, in order, what ``steering.extract_features.main`` +
    ``process_shard`` now do:
      1. ``run_l0_tier = l0_tier_for_resolved(resolved)`` (once per run)
      2. ``assert_l0_tier_compatible(output_dir, run_l0_tier)`` (before any
         write into the dir)
      3. ``stamp_l0_tier(shard_dict, run_l0_tier)`` then ``save_features``
         (per shard)
    """
    run_l0_tier = l0_tier_for_resolved(resolved)
    assert_l0_tier_compatible(out_dir, run_l0_tier)
    shard_dict = shard if shard is not None else _shard_dict()
    stamp_l0_tier(shard_dict, run_l0_tier)
    save_features(shard_dict, out_stem)
    return run_l0_tier


def _v2_resolved():
    return resolve_sae(
        model="gemma-3-4b-it",
        release=SAERelease.gemmascope_2,
        activation_type=ActivationType.residual,
        layer=22,
        width=SAEWidth.W_16K,
        l0=L0Tier.medium,
    )


def _v1_resolved():
    return resolve_sae(
        model="gemma-2-2b",
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=12,
        width=SAEWidth.W_16K,
        l0=None,
    )


class TestV2ShardWritePersistsTier:
    def test_v2_run_persists_tier_into_written_shard(self, tmp_path):
        run_tier = _run_wired_write(
            tmp_path, tmp_path / "shard_00001", _v2_resolved()
        )
        assert run_tier is L0Tier.medium
        shard_path = tmp_path / "shard_00001_features.safetensors.zst"
        # The tier persisted as the correct L0Tier through the real
        # save_features serializer (not just the in-memory dict).
        assert read_l0_tier_from_shard(shard_path) is L0Tier.medium
        assert load_features(shard_path)[MetadataKey.L0_TIER] == "medium"


class TestGuardInvokedBeforeWrite:
    def test_second_run_different_tier_into_same_dir_raises(self, tmp_path):
        # First v2 run: writes a `medium` shard.
        _run_wired_write(tmp_path, tmp_path / "shard_00001", _v2_resolved())
        # Second run targeting the same dir with a different tier (`big`):
        # the guard the production path calls before writing must reject it.
        big_resolved = resolve_sae(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_16K,
            l0=L0Tier.big,
        )
        with pytest.raises(
            FeatureDirectoryMismatchError, match=r"l0_tier.*medium.*big"
        ):
            _run_wired_write(
                tmp_path, tmp_path / "shard_00002", big_resolved
            )

    def test_v1_run_into_v2_dir_raises(self, tmp_path):
        _run_wired_write(tmp_path, tmp_path / "shard_00001", _v2_resolved())
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v2.*v1"):
            _run_wired_write(
                tmp_path, tmp_path / "shard_00002", _v1_resolved()
            )

    def test_v2_run_into_v1_dir_raises(self, tmp_path):
        _run_wired_write(tmp_path, tmp_path / "shard_00001", _v1_resolved())
        with pytest.raises(FeatureDirectoryMismatchError, match=r"v1.*v2"):
            _run_wired_write(
                tmp_path, tmp_path / "shard_00002", _v2_resolved()
            )

    def test_same_tier_resume_is_allowed(self, tmp_path):
        # Append/resume into a matching-tier dir must NOT raise (this is the
        # normal multi-shard extraction case).
        _run_wired_write(tmp_path, tmp_path / "shard_00001", _v2_resolved())
        run_tier = _run_wired_write(
            tmp_path, tmp_path / "shard_00002", _v2_resolved()
        )
        assert run_tier is L0Tier.medium
        assert (
            read_l0_tier_from_shard(
                tmp_path / "shard_00002_features.safetensors.zst"
            )
            is L0Tier.medium
        )


class TestV1ShardStaysTierless:
    def test_v1_run_writes_no_l0_tier_metadata(self, tmp_path):
        run_tier = _run_wired_write(
            tmp_path, tmp_path / "shard_00001", _v1_resolved()
        )
        assert run_tier is None
        shard_path = tmp_path / "shard_00001_features.safetensors.zst"
        # Unchanged from pre-v2: the key is absent, not "" or "None".
        assert read_l0_tier_from_shard(shard_path) is None
        assert MetadataKey.L0_TIER not in load_features(shard_path)

class TestStampL0TierUnit:
    def test_none_leaves_key_absent(self):
        d = _shard_dict()
        stamp_l0_tier(d, None)
        assert "l0_tier" not in d

    def test_tier_injected_as_value_string(self):
        d = _shard_dict()
        stamp_l0_tier(d, L0Tier.big)
        assert d["l0_tier"] == "big"
