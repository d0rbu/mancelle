"""Regression tests for encoder-decoder Hydra composition and CLI contracts.

Issue #722 identified three related failures:

* the documented selection producer path was not declared in the structured
  training config;
* threshold hardware presets recursively merged with the inherited sampling
  mapping, creating an invalid mixed sum-type payload; and
* the torchrun-compatible entrypoint discarded Hydra deletion overrides.

These tests exercise the public command/config surfaces without loading a
model, dataset, or GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import pytest
import yaml
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    _close_training_background_workers,
    _reject_precompute_only_packed_row_dir,
    _resolved_user_config,
    resolve_static_pad_f_from_training_cfg,
)
from steering._common.hydra_cli import parse_config_name_and_overrides
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
    parse_feature_selection,
)
from steering.eval.cache_identity import cache_identity_from_config
from steering.packed_batch_io import compute_selection_hash

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = str((REPO_ROOT / "configs" / "training").resolve())

DEFAULT_FEATURE_SELECTION: dict[str, Any] = {
    "distribution": "geometric",
    "mean": 12.0,
    "std": 12.0,
    "min_k": 1,
    "max_k": 49,
    "delta_weights": "percentile_delta",
    "mode": "top_k_percentile",
}
L22_SPLICE_CF_FEATURE_SELECTION: dict[str, Any] = {
    **DEFAULT_FEATURE_SELECTION,
    # max_k is exclusive, so 49 yields the requested effective maximum of 48.
    "max_k": 49,
}
L22_EVAL_CONTEXT_FEATURE_SELECTION: dict[str, Any] = {
    "distribution": "geometric",
    "mean": 12.0,
    "std": 12.0,
    "min_k": 1,
    "max_k": 49,
    "mode": "top_k_percentile",
}
# D-seed (PR #739 Track D, 2026-07-24) migration: clustered
# eval-context-cache identity is now pinned to the real conditioning.seed
# (42) with delta_weights retained as null (Codex cross-review Finding 1,
# HIGH -- matching the packed-row producer's payload convention) rather
# than a legacy hardcoded seed=0 -- see configs/training/cluster_fp/
# conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml's own migration
# comment. Old (legacy seed=0) value was
# 63aeb505576814764888abb660e9e9154c19bb399541a59bf3f50fc697bcea0f.
L22_FINAL_EVAL_CONTEXT_FEATURE_SELECTION_HASH = (
    "6be4095ad0116d3e1b7ca93a2fe0a47439eaba770c6d0ef08ab164a0652a19bc"
)
THRESHOLD_FEATURE_SELECTION: dict[str, Any] = {
    "threshold_type": "feature_percentile_segment_topk",
    "threshold": 2048.9,
}
DEFAULT_SELECTION_HASH = "49c31e06d6259f38305de0dfb45cb8185786f28839fb10001070fd7f9ec58c54"
SAMPLING_OVERRIDE_SELECTION_HASH = (
    "e6fca8762d07d182b58b413495d7b11d0f2c23bff011d94eec63f61014ffa8f2"
)
PREVIOUS_DEFAULT_SELECTION_HASH = "1b77daf9433c90d94f44034e8fbe2dbf04ff20cb34a7a28df2cb786657645f2c"
H100_SELECTION_HASH = "0ac77df441217e100b5fd928918893fe1994f281bb46717a70262ae9d17d12f4"
THRESHOLD_SELECTION_HASH = "9283f6e087680f4e3957dc6125a03f9a9a071ebbc26d0c010e5a71003fb31f20"

EXPECTED_PRETRAIN_SELECTION_HASHES = {
    "conditional_encdec_train": DEFAULT_SELECTION_HASH,
    "conditional_encdec_train_gemma3": DEFAULT_SELECTION_HASH,
    "conditional_encdec_4090": DEFAULT_SELECTION_HASH,
    "conditional_encdec_4xh200": DEFAULT_SELECTION_HASH,
    "conditional_encdec_simple": DEFAULT_SELECTION_HASH,
    "conditional_encdec_4xh100": H100_SELECTION_HASH,
    "conditional_encdec_d12_baseline_4xh100": H100_SELECTION_HASH,
    "conditional_encdec_d12_fp10_4xh100": H100_SELECTION_HASH,
    "conditional_encdec_d12_fp_4xh100": H100_SELECTION_HASH,
    "conditional_encdec_d12_fp_s32": H100_SELECTION_HASH,
    "conditional_encdec_d12_nextseg_4xh100": H100_SELECTION_HASH,
    "conditional_encdec_delta_4xh100": (
        "f98d09552f48db809d4de226feb90b3d39043bb53fb647689bfe74401dbf325c"
    ),
    "conditional_encdec_delta_8xh100": (
        "f98d09552f48db809d4de226feb90b3d39043bb53fb647689bfe74401dbf325c"
    ),
    "conditional_encdec_splice_cf_gemma3pt_4090": (
        "1e464cc6e9b077842eeb249df7dead951cf8375a19b86eba57ba6796d432c5ab"
    ),
    "conditional_encdec_splice_cf_gemma3pt_8xpro6000": (
        "77edf950afd0f4a9ecc5cddf79d871b7d0329888125772d064a96f3bdff86f91"
    ),
    "conditional_encdec_sweep": (
        "263f2925d11c9c0e58159345831e113b3d33aaac8d4913e5687b4cafa6df6f5e"
    ),
    "conditional_encdec_sweep_4090": (
        "263f2925d11c9c0e58159345831e113b3d33aaac8d4913e5687b4cafa6df6f5e"
    ),
    "bench_live_4090": ("6991948ddc8d80cba78205de703c314580410e793be5cb2118a5895789fd7baa"),
    "bench_prepacked_4090": ("6991948ddc8d80cba78205de703c314580410e793be5cb2118a5895789fd7baa"),
}
THRESHOLD_PRETRAIN_CONFIGS = frozenset(
    {
        "conditional_encdec_2xb200",
        "conditional_encdec_2xh200",
    }
)
SFT_CONFIGS = frozenset(
    {
        "conditional_encdec_sft",
        "conditional_encdec_sft_2xb200",
        "conditional_encdec_sft_2xh200",
        "conditional_encdec_sft_4090",
        "conditional_encdec_sft_4xh200",
        "conditional_encdec_sft_profile_4090",
    }
)
RESIDUAL_HEAD_SFT_CONFIG = "conditional_encdec_sft_3b_residual_head"
TRANSFORMER_HEAD_SFT_CONFIG = "conditional_encdec_sft_3b_transformer_head"
SELFCOND_CONFIGS = frozenset({"conditional_encdec_selfcond"})
TWELVE_FEATURE_SAMPLING_CONFIGS = frozenset(
    {
        "conditional_encdec_train",
        "conditional_encdec_train_gemma3",
        "conditional_encdec_4090",
        "conditional_encdec_4xh200",
        "conditional_encdec_simple",
        "conditional_encdec_splice_cf_gemma3pt_4090",
        "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
    }
)
ALL_PRETRAIN_CONFIGS = frozenset(EXPECTED_PRETRAIN_SELECTION_HASHES) | (THRESHOLD_PRETRAIN_CONFIGS)
SHIPPED_SELECTION_CONFIGS = (
    ALL_PRETRAIN_CONFIGS | SFT_CONFIGS | SELFCOND_CONFIGS
    | {RESIDUAL_HEAD_SFT_CONFIG, TRANSFORMER_HEAD_SFT_CONFIG}
)
# Actual main payloads immediately before issue #1681, independently composed
# from the untouched baseline. Removing the added head/LR keys and the explicit
# warmup/packing fields must recover these ordered payloads for existing SFT presets.
EXPECTED_PRE_RESIDUAL_SFT_CONFIG_SHA256 = {
    "conditional_encdec_sft": "86785ab7e1d987f5bfc378e0c9b9d0939480e30caf1e36caec3ff9b044797764",
    "conditional_encdec_sft_2xb200": "86785ab7e1d987f5bfc378e0c9b9d0939480e30caf1e36caec3ff9b044797764",
    "conditional_encdec_sft_2xh200": "937aeff6adc4d54c357867b95d7407e90b31d84d2953f746c3c8a346dd7b1fbe",
    "conditional_encdec_sft_4090": "1844cd3b5e2ee0c8cc76c3eedf0056540ff9f12532872e3a40f262dab4f87f36",
    "conditional_encdec_sft_4xh200": "e3df6576758f07354c62a9d18544b01915019c50a53a466eef82ee973c4585e6",
    "conditional_encdec_sft_profile_4090": "e14d7b5cc3ebccd417f7e2daf429c0e1565b0e81ac78ba5f950039f1e6f5f90f",
}
EXPECTED_CHECKPOINT_CONFIG_SHA256 = (
    "0b1499bc57f81e2c9c2b3321222b22e4b9b029f1be6b2d3321889bf0b61c6022"
)
EXPECTED_WANDB_CONFIG_SHA256 = "24929d5aac01108d65b6f2697e44a0d824ac1d9536d0fb5f84717f31405906ef"
# D-seed (PR #739 Track D, 2026-07-24) migration: this resolved-config
# whole-JSON hash changed because
# cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml (an
# ancestor default of the composed config below) gained a new
# conditioning.seed: 42 field -- required so clustered eval-context-cache
# identity resolution (steering/eval/cache_identity.py) no longer raises
# -- and its migrated conditioning.identity.feature_selection_hash value
# (Codex cross-review Finding 1, HIGH: corrected from an interim value
# that dropped delta_weights to the packed-row-identity-matching value
# that retains it as null). Original pre-migration value was
# aa67481478a49b0be50b2cae26299aa3401d0cf6109f6122b49044d39f0ed4d3; an
# interim value (before the Finding 1 fix) was
# 17cb280445bf3a821741e196a93566ccce2d58023feccd55da50443970a71128.
# Issues #849 and #853 add operational checkpoint-upload and body-prefetch
# defaults. Both are stripped from checkpoint metadata but intentionally
# remain in this resolved runtime-config hash.
# Issue #1719 adds only the explicit nested mlp/zero-depth architecture to the
# pinned payload; unrelated later runtime-default changes remain unrepinned.
EXPECTED_C1024_REMOTE_PACKED_RESOLVED_CONFIG_SHA256 = (
    "cf7d5c12889cbb3f9b34d8a0044611f62192b37486b7e84cceb71da4fb3486fc"
)
# Issue #1281 splits the shared loop surface into _train_common.yaml. Hydra
# preserves merge/insertion order, so the ordered JSON changes even though the
# dedicated pre-split goldens prove base_train and conditional_encdec_train are
# key/value-identical mappings.
# Issue #1719 updates only head-bearing payloads: nested block/head_layers or
# the canonical SFT override pair. The historical payloads were reconstructed
# and checked against their existing hashes before applying that schema change;
# unrelated save_every and node-cache default drift is deliberately not repinned.
EXPECTED_RESOLVED_CONFIG_SHA256 = {
    "bench_live_4090": "91506263d9337856267c67a85bf154e3c6309febac8af8bd571e58ef7b9ca2cc",
    "bench_prepacked_4090": "70cfcdffe63bb0509194b466db6ac253a8b926dfde7e2c10e1114ae79d5ba528",
    "conditional_encdec_2xb200": "c96f9334503f71b22bdc91db5baefe445cf65123511b2f90fccaea5ccf7a2890",
    "conditional_encdec_2xh200": "bf585beb5b24bb8761e2932618bbb2fa9da1d4698940e4c5644a8af074b5af9e",
    "conditional_encdec_4090": "50777ef10be01c0ddd83b50bc8f2d05ea7266f8df8ee5625a98d753477ca2e1b",
    "conditional_encdec_4xh100": "d13834e4054d5994286149099aace286c3dab055eec7f374335a8dd78b7d64ca",
    "conditional_encdec_4xh200": "0a3e910bc28881967f2edc81ba4566f40e4872a8d3c797c94de9d70ba8a0ced3",
    "conditional_encdec_d12_baseline_4xh100": "92a5ec21635881e9a798e2ac8a49ccb6042654834594a6db8dfe9229d681a67b",
    "conditional_encdec_d12_fp10_4xh100": "71e15dbeb581bde63910cd155118835a48be945085dd2c767b97a6688a3930e4",
    "conditional_encdec_d12_fp_4xh100": "1a8069deee833cf20754c8e53a709083188571c766255e41436393d8f51e23e8",
    "conditional_encdec_d12_fp_s32": "ca4773265663c0bbaeac54265bc6a6973b7409cbcfa6b674b4f17b03fb8e83c7",
    "conditional_encdec_d12_nextseg_4xh100": "d5c4f80a5053b157fae8e960f83cd6ad5867c29372c443ac9290fcc529e9e917",
    "conditional_encdec_delta_4xh100": "710a01936775821ca689fe9b8d563ee3098cca88b77fd3bec3b0168e1851ebce",
    "conditional_encdec_delta_8xh100": "be84a581c77e37b0fa60e61182cef44ea9f1c2317f76c605a55d73aab0543931",
    # Conditioning fidelity vs pretraining remains hardcoded policy, not a
    # config knob (see steering/conditional/checkpoint_conditioning.py).
    # Two independent additions moved these hashes and both are present here:
    # issue #1475 adds one SFT execution key (resume-safe batch-prefetch
    # depth), and the SFT self-conditioning data source adds six selfcond keys
    # (data_source, sft_dataset, sft_feature_dir, sft_parquet_dir,
    # val_sft_feature_dir, val_sft_parquet_dir). The two are disjoint: the SFT
    # hashes below are main's (self-conditioning changed no SFT config key),
    # and only ``conditional_encdec_selfcond`` carries the six new keys.
    # The SFT family's ``compile_static`` default moved ``false`` -> ``"auto"``
    # (dynamic=None auto-dynamic: concrete decoder dims for the torch>=2.13
    # flex FLASH backend, symbolic mark_dynamic'd packed extents).  Every SFT
    # hash below moved together; the hardware overlays inherit the base value.
    # They moved together again for the steered SFT mixtures, which add exactly
    # one nullable SFT key (``steered_jsonl``) to the base config: stripping
    # that one key from each composed payload reproduces the previous hash.
    # Issue #1681 introduced feature_prediction_mlp_layers=0. Issue #1719
    # replaces it with block=mlp and head_layers=0; stripping the head defaults
    # retains the separate pre-residual golden contract above.
    # PR #1726 adds fp_head_lr=null after unembedding_lr and issue #1735 adds
    # fp_head_matrix_lr=0.02 beside it. Each historical payload was checked
    # against its old hash before adding those fields; unrelated save_every
    # drift remains deliberately unrepinned.
    "conditional_encdec_sft": "61438c4d6c517d79ba47f66c4fc9d3337aaeb3a2bd0bab0c6acec131ba18ad5f",
    "conditional_encdec_sft_2xb200": "61438c4d6c517d79ba47f66c4fc9d3337aaeb3a2bd0bab0c6acec131ba18ad5f",
    "conditional_encdec_sft_2xh200": "bebe8ebcdf85d32e3b56ef83aa6f8bc6eee9c0cc89eacf5f024b6ecfca7e3d86",
    "conditional_encdec_sft_4090": "c54ce5e99b6c42a2d566aae42d1380440ad2c07db1fe8341a52bae9b4fe451ff",
    "conditional_encdec_sft_4xh200": "c89c3f4be28519b82e79c45a74cfca4f9f8fee5c07f66cd0506da086f691f355",
    "conditional_encdec_sft_profile_4090": "f7c113615c73c3993f6d66f5ced94d554db5a983c20e08f35432accb99083ba0",
    RESIDUAL_HEAD_SFT_CONFIG: "0ccbd927cde029a613cdc0991dcf3a13d018db5c0139c98467a03bab80b9df60",
    TRANSFORMER_HEAD_SFT_CONFIG: "ade1fe54abc2f4e0440ca48e3ceb1e2488feb1b80c1336d0b08a475486bf1bac",
    "conditional_encdec_selfcond": "492815f63fa03d8bffe4b7099c52a84554b16220e0c9db643c6e73338b1c763c",
    "conditional_encdec_simple": "d63ad89f43b7a48fd7f717acb0cd5561fd1070730264f3248cd872bbc615873a",
    "conditional_encdec_splice_cf_gemma3pt_4090": "f8a091055aed9964eafb44c45dc23489be4ff26cc64f50d1bbde54788ee15057",
    "conditional_encdec_splice_cf_gemma3pt_8xpro6000": "6819bdff71c63053ea5e6caf16919c396ac8abca3be6415e86f41cb7618db9fe",
    "conditional_encdec_sweep": "1249adcb64e8ab41886c0ea3a862c26a13474f76c97850a205b415e23ad25a65",
    "conditional_encdec_sweep_4090": "2d6d747ad8884f4653f5cb9ba3373c750e161e85d79c2df3e306da2d2044ded1",
    "conditional_encdec_train": "dd8cdc70a145d829d10675e40af19c5c7d1360399423a67dd5da7e5f4e4ec3bb",
    "conditional_encdec_train_gemma3": "8ba5eab2a7fc9dc85f1c3b15e3d4618120fff3441ea11d7dc397bf727ee52572",
}


@pytest.fixture
def _residual_head_artifact_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the production recipe without depending on operator paths."""
    for suffix in (
        "RUN_DIR",
        "STEERED_JSONL",
        "TRAIN_FEATURES",
        "VAL_FEATURES",
        "TRAIN_PARQUET",
        "VAL_PARQUET",
        "MASK",
        "CLUSTERS",
    ):
        monkeypatch.setenv(f"RESIDUAL_HEAD_{suffix}", f"/fixture/residual-head/{suffix.lower()}")


def _as_dict(value: Any) -> dict[str, Any]:
    raw = OmegaConf.to_container(value, resolve=True)
    assert isinstance(raw, dict)
    return cast(dict[str, Any], raw)


def _selection_hash(cfg: Any) -> str:
    negative_selection = (
        _as_dict(cfg.negative_selection) if cfg.get("negative_selection") is not None else None
    )
    feature_delta = _as_dict(cfg.feature_delta) if cfg.get("feature_delta") is not None else None
    return compute_selection_hash(
        positive_selection=_as_dict(cfg.feature_selection),
        negative_selection=negative_selection,
        delta_config=feature_delta,
        num_buckets=int(cfg.num_buckets),
        seed=int(cfg.seed),
        ddp_rank=0,
        splice_batch_size=cfg.get("selection_splice_batch_size"),
    )


def _ordered_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_default_feature_selection_payload_and_hash_are_pinned(
    hydra_compose,
) -> None:
    """The project default is the requested 12/12/48 delta-ranked arm."""
    cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    feature_selection = _as_dict(cfg.feature_selection)
    negative_selection = _as_dict(cfg.negative_selection)
    feature_delta = _as_dict(cfg.feature_delta)

    assert feature_selection == DEFAULT_FEATURE_SELECTION
    parsed = parse_feature_selection(feature_selection)
    assert isinstance(parsed, SamplingSelection)
    assert parsed.delta_weights is DeltaSamplingWeights.percentile_delta
    assert (
        compute_selection_hash(
            positive_selection=feature_selection,
            negative_selection=negative_selection,
            delta_config=feature_delta,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.seed),
            ddp_rank=0,
        )
        == DEFAULT_SELECTION_HASH
    )


def test_c1024_remote_training_config_is_explicit_and_opt_in(hydra_compose):
    base_cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    assert base_cfg.remote_training is None

    cfg = hydra_compose(
        CONFIG_DIR,
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote",
    )
    assert cfg.max_seq_len == 1024
    assert cfg.use_precomputed_batches is True
    assert cfg.feature_clusters is not None
    assert cfg.feature_delta.mode == "splice_counterfactual"
    assert cfg.feature_selection.min_k == 1
    assert cfg.feature_selection.max_k == 49
    assert cfg.negative_selection.min_k == 0
    assert cfg.negative_selection.max_k == 49
    assert cfg.conditioning.feature_selection.min_k == 1
    assert cfg.conditioning.feature_selection.max_k == 49
    assert cfg.conditioning.negative_selection.min_k == 0
    assert cfg.conditioning.negative_selection.max_k == 49
    assert cfg.selection_splice_batch_size == 64
    assert cfg.model_tag == "l22_gemma3pt_clustered_splice_cf_c1024_remote_v1_781"
    assert OmegaConf.to_container(cfg.remote_training, resolve=True) == {
        "bucket_id": "d0rb/nano-gpt-pretrain-steer-artifacts",
        "body_prefetch": {
            "lookahead_shards_per_rank": 2,
        },
        "selection_prefix": "features",
        "checkpoint_prefix": "checkpoints/conditional_encdec",
        "selection_wait_timeout_seconds": 14400,
        "checkpoint_retry_timeout_seconds": 1800,
        "retry_interval_seconds": 30,
        "resume_latest": True,
        "async_checkpoint_upload": True,
    }
    sync_cfg = hydra_compose(
        CONFIG_DIR,
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote",
        overrides=["remote_training.async_checkpoint_upload=false"],
    )
    assert sync_cfg.remote_training.async_checkpoint_upload is False
    assert _resolved_user_config(cfg)["remote_training"]["async_checkpoint_upload"] is True
    assert _resolved_user_config(sync_cfg)["remote_training"]["async_checkpoint_upload"] is False
    assert _checkpoint_user_config(cfg) == _checkpoint_user_config(sync_cfg)
    assert "body_prefetch" not in _checkpoint_user_config(cfg)["remote_training"]
    assert _checkpoint_user_config(cfg)["remote_training"] == {
        "bucket_id": "d0rb/nano-gpt-pretrain-steer-artifacts",
        "selection_prefix": "features",
        "checkpoint_prefix": "checkpoints/conditional_encdec",
        "selection_wait_timeout_seconds": 14400,
        "checkpoint_retry_timeout_seconds": 1800,
        "retry_interval_seconds": 30,
        "resume_latest": True,
    }
    packed_cfg = hydra_compose(
        CONFIG_DIR,
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed",
    )
    assert packed_cfg.run != cfg.run
    assert packed_cfg.model_tag != cfg.model_tag
    assert packed_cfg.model_tag == ("l22_gemma3pt_clustered_splice_cf_c1024_remote_packed_v1_806")
    assert packed_cfg.steering_eval_every == 250
    assert OmegaConf.to_container(
        packed_cfg.remote_training,
        resolve=True,
    ) == {
        **_as_dict(cfg.remote_training),
        "body_prefetch": {
            "lookahead_shards_per_rank": 2,
            "cache_shards": 24,
        },
        "train_packed": {
            "trust_existing_cache": True,
        },
    }
    assert _ordered_json_sha256(_as_dict(packed_cfg)) == (
        EXPECTED_C1024_REMOTE_PACKED_RESOLVED_CONFIG_SHA256
    )


def test_no_shipped_positive_sampling_config_allows_zero_features() -> None:
    """Every active positive-count config must request at least one feature."""
    positive_keys = {"feature_selection", "positive_selection", "positive_counts"}
    violations: list[str] = []

    def visit(value: Any, *, path: tuple[str, ...], source: Path) -> None:
        if isinstance(value, dict):
            if (
                path
                and path[-1] in positive_keys
                and "distribution" in value
                and value.get("min_k") == 0
            ):
                violations.append(f"{source}:{'.'.join(path)}.min_k")
            for key, child in value.items():
                visit(child, path=(*path, str(key)), source=source)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path=(*path, str(index)), source=source)

    for source in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        document = yaml.safe_load(source.read_text())
        initial_path = (
            ("feature_selection",) if source.parent.name == "feature_selection_variant" else ()
        )
        visit(document, path=initial_path, source=source.relative_to(REPO_ROOT))

    assert violations == []


def test_sampling_override_inherits_the_new_delta_weight_contract(
    hydra_compose,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[
            "feature_selection.mean=7.0",
            "feature_selection.std=7.0",
            "feature_selection.max_k=29",
        ],
    )
    feature_selection = _as_dict(cfg.feature_selection)

    assert feature_selection == {
        **DEFAULT_FEATURE_SELECTION,
        "mean": 7.0,
        "std": 7.0,
        "max_k": 29,
    }
    assert (
        compute_selection_hash(
            positive_selection=feature_selection,
            negative_selection=_as_dict(cfg.negative_selection),
            delta_config=_as_dict(cfg.feature_delta),
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.seed),
            ddp_rank=0,
        )
        == SAMPLING_OVERRIDE_SELECTION_HASH
    )


def test_previous_default_remains_exactly_recoverable_with_explicit_overrides(
    hydra_compose,
) -> None:
    """The requested new default does not remove the old sampling contract."""
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[
            "feature_selection.mean=5.0",
            "feature_selection.std=5.0",
            "feature_selection.max_k=25",
            "feature_selection.delta_weights=absolute_percentile",
        ],
    )
    feature_selection = _as_dict(cfg.feature_selection)

    assert feature_selection == {
        "distribution": "geometric",
        "mean": 5.0,
        "std": 5.0,
        "min_k": 1,
        "max_k": 25,
        "delta_weights": "absolute_percentile",
        "mode": "top_k_percentile",
    }
    assert _selection_hash(cfg) == PREVIOUS_DEFAULT_SELECTION_HASH


@pytest.mark.parametrize("config_name", sorted(TWELVE_FEATURE_SAMPLING_CONFIGS))
def test_requested_sampling_default_reaches_every_inheriting_pretraining_preset(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    expected = (
        L22_SPLICE_CF_FEATURE_SELECTION
        if config_name == "conditional_encdec_splice_cf_gemma3pt_8xpro6000"
        else DEFAULT_FEATURE_SELECTION
    )
    assert _as_dict(cfg.feature_selection) == expected


def test_l22_training_config_pins_the_eval_cache_identity_for_its_final_policy(
    hydra_compose,
) -> None:
    """Keep the training preset pinned to its exact eval-context cache leaf.

    ``resolve_feature_selection_hash`` recomputes this value from the resolved
    conditioning policy, whitelist, and cluster descriptor at cache build and
    train startup.  This golden catches accidental config drift before either
    path can consume a mismatched cache.
    """

    cfg = hydra_compose(
        CONFIG_DIR,
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000",
    )
    assert _as_dict(cfg.conditioning.feature_selection) == L22_EVAL_CONTEXT_FEATURE_SELECTION
    assert _as_dict(cfg.conditioning.negative_selection) == {
        "distribution": "geometric",
        "mean": 12.0,
        "min_k": 0,
        "max_k": 49,
    }
    assert (
        cfg.conditioning.identity.feature_selection_hash
        == L22_FINAL_EVAL_CONTEXT_FEATURE_SELECTION_HASH
    )


def test_l22_training_config_clustered_identity_requires_and_resolves_seed(
    hydra_compose,
    tmp_path,
) -> None:
    """D-seed (PR #739 Track D) regression guard -- fixture-level, no heavy
    real assets (unlike tests/steering/test_issue768_production_artifacts.py's
    @slow/@requires_feature_data real-artifact test, which needs the actual
    ~6.6MB mask + ~150KB clusters + multi-GB feature-statistics files).

    The shipped clustered config configs/training/cluster_fp/
    conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml must carry
    conditioning.seed, or cache_identity_from_config
    (steering/eval/cache_identity.py's clustered branch) raises before any
    cache I/O. This is the exact crash D-seed review Finding 1 (HIGH) caught
    against _build_cached_context_provider -- the real training-loop call
    path (conditioning.context_cache_dir is set, so this function runs on
    every training launch of this config).
    """
    cfg = hydra_compose(
        CONFIG_DIR,
        "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000",
    )
    # Positive: the shipped config carries a real seed today.
    assert cfg.conditioning.get("seed") is not None

    # Tiny synthetic mask + cluster fixtures (not the real production
    # artifacts) so this test needs no external data and runs in any
    # environment.
    mask_path = tmp_path / "mask.txt"
    mask_path.write_text("\n".join(str(i) for i in range(8)), encoding="utf-8")
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 3], [2, 4]]}),
        encoding="utf-8",
    )
    cfg.conditioning.feature_mask.path = str(mask_path)
    cfg.conditioning.feature_clusters.path = str(cluster_path)
    cfg.conditioning.n_sae_features = 8
    # Clear the pinned production hash -- this test only proves the
    # clustered branch resolves SOME hash without crashing on the seed
    # check, not that it matches a specific production value (that is
    # test_issue768_production_artifacts.py's job, against the real
    # artifacts).
    cfg.conditioning.identity.feature_selection_hash = None

    identity = cache_identity_from_config(cfg.conditioning.identity, cfg.conditioning)
    assert identity.feature_selection_hash is not None

    # Negative: removing conditioning.seed must raise loudly, not silently
    # fall back to a default -- the exact regression class this test
    # guards against. Hydra-composed configs are struct-mode by default
    # (deletion disallowed); relax it only for this one key removal.
    OmegaConf.set_struct(cfg.conditioning, False)
    del cfg.conditioning["seed"]
    with pytest.raises(ValueError, match="conditioning.seed"):
        cache_identity_from_config(cfg.conditioning.identity, cfg.conditioning)


def test_delta_weight_default_does_not_implicitly_enable_counterfactual_splicing(
    hydra_compose,
) -> None:
    default_cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    splice_cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_splice_cf_gemma3pt_4090",
    )

    assert default_cfg.feature_selection.delta_weights == "percentile_delta"
    assert default_cfg.feature_delta.mode == "prev_segment"
    assert splice_cfg.feature_selection.delta_weights == "percentile_delta"
    assert splice_cfg.feature_delta.mode == "splice_counterfactual"


@pytest.mark.parametrize(
    ("config_name", "expected_pad_f"),
    [
        ("conditional_encdec_train", 64),
        ("conditional_encdec_4090", 64),
        ("conditional_encdec_4xh200", 64),
        ("conditional_encdec_splice_cf_gemma3pt_4090", 64),
        ("conditional_encdec_splice_cf_gemma3pt_8xpro6000", 98),
        ("conditional_encdec_2xh200", None),
    ],
)
def test_requested_max_k_propagates_to_static_padding_contract(
    hydra_compose,
    config_name: str,
    expected_pad_f: int | None,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    assert resolve_static_pad_f_from_training_cfg(cfg) == expected_pad_f


def test_requested_default_uses_a_distinct_checkpoint_path_component(
    hydra_compose,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    current = parse_feature_selection(_as_dict(cfg.feature_selection))
    assert feature_selection_dirname(current) == "dist_geometric/mean_12.0/std_12.0"

    old_cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[
            "feature_selection.mean=5.0",
            "feature_selection.std=5.0",
            "feature_selection.max_k=25",
            "feature_selection.delta_weights=absolute_percentile",
        ],
    )
    previous = parse_feature_selection(_as_dict(old_cfg.feature_selection))
    assert feature_selection_dirname(previous) == "dist_geometric/mean_5.0/std_5.0"


def test_representative_inherited_sampling_preset_is_unchanged(
    hydra_compose,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_4xh100")
    feature_selection = _as_dict(cfg.feature_selection)

    assert feature_selection == {
        "distribution": "geometric",
        "mean": 10.0,
        "std": 10.0,
        "min_k": 1,
        "max_k": 40,
        "delta_weights": None,
        "mode": "top_k_percentile",
    }
    assert (
        compute_selection_hash(
            positive_selection=feature_selection,
            negative_selection=_as_dict(cfg.negative_selection),
            delta_config=None,
            num_buckets=int(cfg.num_buckets),
            seed=int(cfg.seed),
            ddp_rank=0,
        )
        == H100_SELECTION_HASH
    )


@pytest.mark.parametrize(
    ("config_name", "expected_hash"),
    sorted(EXPECTED_PRETRAIN_SELECTION_HASHES.items()),
)
def test_every_pretraining_selection_hash_matches_the_pinned_contract(
    hydra_compose,
    config_name: str,
    expected_hash: str,
) -> None:
    """Goldens pin requested default drift and unchanged explicit presets."""
    cfg = hydra_compose(CONFIG_DIR, config_name)
    assert parse_feature_selection(_as_dict(cfg.feature_selection)) is not None
    assert _selection_hash(cfg) == expected_hash


@pytest.mark.parametrize("config_name", sorted(SFT_CONFIGS))
def test_sft_feature_selection_payloads_are_unchanged(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    assert _as_dict(cfg.feature_selection) == {
        "distribution": "geometric",
        "mean": 10.0,
        "std": 10.0,
        "min_k": 1,
        "max_k": 40,
        "mode": "top_k_percentile",
        "delta_weights": None,
    }


@pytest.mark.parametrize(
    ("config_name", "expected_legacy_sha256"),
    sorted(EXPECTED_PRE_RESIDUAL_SFT_CONFIG_SHA256.items()),
)
def test_sft_head_defaults_are_the_only_ordered_payload_change(
    hydra_compose,
    config_name: str,
    expected_legacy_sha256: str,
) -> None:
    payload = _as_dict(hydra_compose(CONFIG_DIR, config_name))
    warmup = payload.pop("lr_warmup_steps")
    assert type(warmup) is int
    assert warmup == (0 if config_name == "conditional_encdec_sft_profile_4090" else 20)
    assert payload.pop("muon_parameter_packing") is False
    assert payload.pop("feature_prediction_block") == "mlp"
    depth = payload.pop("feature_prediction_head_layers")
    assert type(depth) is int and depth == 0
    assert payload.pop("fp_head_lr") is None
    assert payload.pop("fp_head_matrix_lr") == 0.02
    assert _ordered_json_sha256(payload) == expected_legacy_sha256


@pytest.mark.parametrize(
    "config_name",
    ["conditional_encdec_2xb200", "conditional_encdec_2xh200"],
)
def test_threshold_hardware_presets_replace_the_sampling_arm(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    raw = _as_dict(cfg.feature_selection)

    assert raw == THRESHOLD_FEATURE_SELECTION
    parsed = parse_feature_selection(raw)
    assert isinstance(parsed, ThresholdSelection)
    assert parsed.threshold_type.value == "feature_percentile_segment_topk"
    assert parsed.threshold == 2048.9
    assert _selection_hash(cfg) == THRESHOLD_SELECTION_HASH


def test_delete_then_add_override_replaces_feature_selection_sum_type(
    hydra_compose,
) -> None:
    """The shared CLI parser and real Hydra config must compose together."""
    argv = [
        "--",
        "config_name=conditional_encdec_train",
        "~feature_selection",
        "+feature_selection.threshold_type=feature_percentile",
        "+feature_selection.threshold=0.9",
        "++seed=7",
        "--local-rank=0",
    ]
    config_name, overrides = parse_config_name_and_overrides(
        argv,
        default_config_name="conditional_encdec_train",
    )

    assert config_name == "conditional_encdec_train"
    assert overrides == [
        "~feature_selection",
        "+feature_selection.threshold_type=feature_percentile",
        "+feature_selection.threshold=0.9",
        "++seed=7",
    ]

    cfg = hydra_compose(CONFIG_DIR, config_name, overrides=overrides)
    assert _as_dict(cfg.feature_selection) == {
        "threshold_type": "feature_percentile",
        "threshold": 0.9,
    }
    assert cfg.seed == 7
    assert isinstance(parse_feature_selection(_as_dict(cfg.feature_selection)), ThresholdSelection)


def test_internal_feature_selection_config_group_composes_clean_arms(
    hydra_compose,
) -> None:
    threshold_cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=["feature_selection_variant=threshold_2048p9"],
    )
    sampling_cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_2xh200",
        overrides=["feature_selection_variant=sampling"],
    )

    assert _as_dict(threshold_cfg.feature_selection) == THRESHOLD_FEATURE_SELECTION
    assert _as_dict(sampling_cfg.feature_selection) == DEFAULT_FEATURE_SELECTION


@pytest.mark.parametrize("config_name", sorted(ALL_PRETRAIN_CONFIGS))
def test_delete_then_add_replaces_selection_from_every_pretraining_preset(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        config_name,
        overrides=[
            "~feature_selection",
            "+feature_selection.threshold_type=feature_percentile",
            "+feature_selection.threshold=0.9",
        ],
    )
    raw = _as_dict(cfg.feature_selection)
    assert raw == {
        "threshold_type": "feature_percentile",
        "threshold": 0.9,
    }
    assert isinstance(parse_feature_selection(raw), ThresholdSelection)


def test_training_entrypoint_forwards_deletion_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the actual ``if __name__ == "__main__"`` argv-routing block.

    ``compose`` deliberately stops execution before any model/data startup.
    This catches a future regression where the entrypoint bypasses the shared
    helper even if the helper's own unit tests remain green.
    """

    class ComposeReached(RuntimeError):
        pass

    captured: dict[str, Any] = {}

    def fake_compose(*, config_name: str, overrides: list[str]):
        captured["config_name"] = config_name
        captured["overrides"] = overrides
        raise ComposeReached

    monkeypatch.setattr(hydra, "compose", fake_compose)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.train.conditional_encdec_train",
            "--",
            "config_name=conditional_encdec_train",
            "~feature_selection",
            "+feature_selection.threshold_type=feature_percentile",
            "+feature_selection.threshold=0.9",
            "--local-rank=0",
        ],
    )
    monkeypatch.delitem(
        sys.modules,
        "scripts.train.conditional_encdec_train",
        raising=False,
    )

    with pytest.raises(ComposeReached):
        runpy.run_module(
            "scripts.train.conditional_encdec_train",
            run_name="__main__",
        )

    assert captured == {
        "config_name": "conditional_encdec_train",
        "overrides": [
            "~feature_selection",
            "+feature_selection.threshold_type=feature_percentile",
            "+feature_selection.threshold=0.9",
        ],
    }


def test_training_entrypoint_uses_project_owned_hydra_helper() -> None:
    source = (REPO_ROOT / "scripts" / "train" / "conditional_encdec_train.py").read_text()
    assert "from steering._common.hydra_cli import" in source
    assert "parse_config_name_and_overrides" in source
    assert 'if "=" in a and not a.startswith("-")' not in source, (
        "conditional_encdec_train.py still has the old inline parser"
    )


def test_precompute_documented_packed_row_override_composes(
    tmp_path: Path,
) -> None:
    """Exercise the public clustered-preselection spelling shown in the docs."""
    cluster_path = tmp_path / "clusters.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.preprocess.precompute_selected_features",
            f"packed_row_dir={tmp_path}",
            f"+feature_clusters.path={cluster_path}",
            "feature_delta=null",
            "feature_selection.delta_weights=null",
            "--cfg",
            "job",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert f"packed_row_dir: {tmp_path}" in result.stdout
    assert f"path: {cluster_path}" in result.stdout
    assert "feature_delta: null" in result.stdout
    assert "delta_weights: null" in result.stdout
    assert "Key 'packed_row_dir' is not in struct" not in combined


def test_precompute_missing_required_path_reaches_application_error() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.preprocess.precompute_selected_features",
            "feature_delta=null",
            "feature_selection.delta_weights=null",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert ("packed_row_dir is required (pass packed_row_dir=path/to/packed_rows)") in combined


def test_documented_cluster_producer_commands_disable_delta_weights() -> None:
    paths = [
        REPO_ROOT / "scripts" / "preprocess" / "precompute_selected_features.py",
        REPO_ROOT / "docs" / "cli-reference.md",
        REPO_ROOT / "docs" / "pipelines" / "encoder-decoder-training.md",
        REPO_ROOT / "docs" / "reference" / "file-reference.md",
    ]
    for path in paths:
        text = path.read_text()
        assert "packed_row_dir=" in text
        assert "feature_delta=null" in text
        assert "feature_selection.delta_weights=null" in text
        assert "+packed_row_dir=" not in text


def test_documented_gradient_training_overrides_compose_a_valid_pipeline(
    hydra_compose,
) -> None:
    overrides = [
        "feature_delta=null",
        "feature_selection.delta_weights=null",
        "+feature_selection.gradient.mode=future_half_split",
    ]
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=overrides,
    )
    parsed = parse_feature_selection(_as_dict(cfg.feature_selection))

    assert cfg.feature_delta is None
    assert isinstance(parsed, SamplingSelection)
    assert parsed.delta_weights is None
    assert parsed.gradient is not None
    assert parsed.gradient.mode.value == "future_half_split"

    gradient_doc = (
        REPO_ROOT / "docs" / "pipelines" / "gradient-attribution-selection.md"
    ).read_text()
    command_start = gradient_doc.index(
        "torchrun --standalone --nproc_per_node=4 -m scripts.train.conditional_encdec_train --"
    )
    command_end = gradient_doc.index("```", command_start)
    command = gradient_doc[command_start:command_end]
    assert "feature_delta=null" in command
    assert "feature_selection.delta_weights=null" in command
    assert "'+feature_selection.gradient.mode=future_half_split'" in command


def test_precompute_packed_row_override_reaches_application_validation(
    tmp_path: Path,
) -> None:
    missing_dir = tmp_path / "missing-packed-rows"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.preprocess.precompute_selected_features",
            f"packed_row_dir={missing_dir}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert f"packed_row_dir not found: {missing_dir}" in combined
    assert "Key 'packed_row_dir' is not in struct" not in combined


@pytest.mark.parametrize(
    "config_name",
    [
        "conditional_encdec_train",
        "conditional_encdec_2xb200",
        "conditional_encdec_2xh200",
        "conditional_encdec_train_gemma3",
    ],
)
def test_packed_row_dir_is_declared_across_training_config_inheritance(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    assert "packed_row_dir" in cfg
    assert cfg.packed_row_dir is None


@pytest.mark.parametrize(
    "override_prefix",
    ["", "+", "++"],
    ids=["ordinary", "append", "force-add"],
)
def test_packed_row_dir_override_spellings_resolve_identically(
    hydra_compose,
    tmp_path: Path,
    override_prefix: str,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[f"{override_prefix}packed_row_dir={tmp_path}"],
    )
    assert cfg.packed_row_dir == str(tmp_path)


def test_unknown_packed_row_dir_typo_still_fails(
    hydra_compose,
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigCompositionException, match="packed_rows_dir"):
        hydra_compose(
            CONFIG_DIR,
            "conditional_encdec_train",
            overrides=[f"packed_rows_dir={tmp_path}"],
        )


def test_unknown_delete_override_is_forwarded_and_fails_loudly(
    hydra_compose,
) -> None:
    config_name, overrides = parse_config_name_and_overrides(
        ["~not_a_real_config_key"],
        default_config_name="conditional_encdec_train",
    )
    assert overrides == ["~not_a_real_config_key"]
    with pytest.raises(ConfigCompositionException, match="not_a_real_config_key"):
        hydra_compose(CONFIG_DIR, config_name, overrides=overrides)


@pytest.mark.parametrize("operator", ["+", "++"])
def test_added_feature_selection_typo_fails_at_application_parse(
    hydra_compose,
    operator: str,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[f"{operator}feature_selection.typo=1"],
    )
    with pytest.raises(ValueError, match=r"unknown or incompatible.*typo"):
        parse_feature_selection(_as_dict(cfg.feature_selection))


def test_cluster_vector_mean_override_composes_on_the_training_config(
    hydra_compose,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=["initialize_encoder_embeddings_cluster_vectors=mean"],
    )
    assert cfg.initialize_encoder_embeddings_cluster_vectors == "mean"


def test_background_worker_teardown_accepts_no_workers() -> None:
    _close_training_background_workers(
        cuda_staging_loader=None,
        checkpoint_publisher=None,
        active_error=None,
    )


def test_background_worker_teardown_uses_strict_remote_order() -> None:
    calls: list[str] = []

    class Closer:
        def __init__(self, label: str) -> None:
            self.label = label

        def close(self) -> None:
            calls.append(self.label)

        def is_terminal(self) -> bool:
            return True

    class BodyWorker:
        def stop_worker(self) -> None:
            calls.append("body-worker")

        def worker_is_terminal(self) -> bool:
            return True

        def release_session(self) -> None:
            calls.append("session-lock")

    _close_training_background_workers(
        batch_prefetch_loader=cast(Any, Closer("batch")),
        cuda_staging_loader=cast(Any, Closer("cuda")),
        remote_body_prefetcher=cast(Any, BodyWorker()),
        checkpoint_publisher=cast(Any, Closer("checkpoint")),
        active_error=None,
    )

    assert calls == [
        "batch",
        "cuda",
        "body-worker",
        "checkpoint",
        "session-lock",
    ]


def test_background_worker_teardown_preserves_primary_error_and_closes_both() -> None:
    calls: list[str] = []

    class FailingCloser:
        def __init__(self, label: str) -> None:
            self.label = label

        def close(self) -> None:
            calls.append(self.label)
            raise OSError(f"{self.label}-failed")

        def is_terminal(self) -> bool:
            return True

    primary = RuntimeError("training-failed")
    _close_training_background_workers(
        cuda_staging_loader=cast(Any, FailingCloser("staging")),
        checkpoint_publisher=cast(Any, FailingCloser("publisher")),
        active_error=primary,
    )

    assert calls == ["staging", "publisher"]
    assert primary.__notes__ == [
        "CUDA staging loader also failed during teardown: OSError: staging-failed",
        "remote checkpoint publisher also failed during teardown: OSError: publisher-failed",
    ]


def test_background_worker_teardown_raises_unmasked_failure_after_all_closes() -> None:
    calls: list[str] = []

    class Closer:
        def __init__(self, label: str, *, fail: bool) -> None:
            self.label = label
            self.fail = fail

        def close(self) -> None:
            calls.append(self.label)
            if self.fail:
                raise OSError(f"{self.label}-failed")

        def is_terminal(self) -> bool:
            return True

    with pytest.raises(OSError, match="staging-failed") as exc_info:
        _close_training_background_workers(
            cuda_staging_loader=cast(Any, Closer("staging", fail=True)),
            checkpoint_publisher=cast(Any, Closer("publisher", fail=True)),
            active_error=None,
        )
    assert calls == ["staging", "publisher"]
    assert exc_info.value.__notes__ == [
        "remote checkpoint publisher also failed during teardown: OSError: publisher-failed"
    ]


def test_checkpoint_user_config_omits_producer_only_packed_row_dir(
    hydra_compose,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, "conditional_encdec_train")
    wandb_config = _resolved_user_config(cfg)
    checkpoint_config = _checkpoint_user_config(cfg)

    assert "packed_row_dir" not in wandb_config
    assert "packed_row_dir" not in checkpoint_config
    assert "selection_splice_batch_size" not in wandb_config
    assert "selection_splice_batch_size" not in checkpoint_config
    for producer_only_key in (
        "selection_source_cache_shards",
        "selection_sidecar_cache_shards",
        "selection_num_workers",
        "selection_resume",
        "selection_packed_spool_dir",
        "selection_shared_store",
        "selection_shared_store_capacity",
        "remote_preselection",
    ):
        assert producer_only_key not in wandb_config
        assert producer_only_key not in checkpoint_config
    assert wandb_config["feature_clusters"] is None
    assert "feature_clusters" not in checkpoint_config
    assert wandb_config["initialize_encoder_embeddings_cluster_vectors"] == "mean"
    assert "initialize_encoder_embeddings_cluster_vectors" not in checkpoint_config
    assert _ordered_json_sha256(wandb_config) == EXPECTED_WANDB_CONFIG_SHA256
    assert _ordered_json_sha256(checkpoint_config) == EXPECTED_CHECKPOINT_CONFIG_SHA256


def test_default_splice_batch_size_is_retained_as_semantic_artifact_identity(
    hydra_compose,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
    )
    assert cfg.selection_splice_batch_size == 64
    assert _resolved_user_config(cfg)["selection_splice_batch_size"] == 64
    assert _checkpoint_user_config(cfg)["selection_splice_batch_size"] == 64


def test_explicit_legacy_splice_batch_size_preserves_checkpoint_metadata_shape(
    hydra_compose,
) -> None:
    cfg = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
        overrides=["selection_splice_batch_size=1"],
    )
    assert cfg.selection_splice_batch_size == 1
    assert "selection_splice_batch_size" not in _resolved_user_config(cfg)
    assert "selection_splice_batch_size" not in _checkpoint_user_config(cfg)


def test_explicit_splice_batch_size_changes_only_splice_selection_hash(
    hydra_compose,
) -> None:
    legacy_splice = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
        overrides=["selection_splice_batch_size=1"],
    )
    batched_splice = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_splice_cf_gemma3pt_8xpro6000",
    )
    assert _selection_hash(legacy_splice) != _selection_hash(batched_splice)

    legacy_delta_free = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=[
            "feature_delta=null",
            "selection_splice_batch_size=1",
        ],
    )
    batched_delta_free = hydra_compose(
        CONFIG_DIR,
        "conditional_encdec_train",
        overrides=["feature_delta=null"],
    )
    assert _selection_hash(legacy_delta_free) == _selection_hash(batched_delta_free)
    assert "selection_splice_batch_size" not in _resolved_user_config(batched_delta_free)
    assert "selection_splice_batch_size" not in _checkpoint_user_config(batched_delta_free)


def test_training_rejects_non_null_producer_only_packed_row_dir() -> None:
    _reject_precompute_only_packed_row_dir(OmegaConf.create({"packed_row_dir": None}))
    with pytest.raises(
        ValueError,
        match="only consumed by scripts.preprocess.precompute_selected_features",
    ):
        _reject_precompute_only_packed_row_dir(OmegaConf.create({"packed_row_dir": "/tmp/packed"}))


def test_fast_sweep_uses_sanitized_user_config_snapshot() -> None:
    source = (REPO_ROOT / "scripts" / "train" / "run_encdec_sweep_fast.py").read_text()
    assert "_resolved_user_config" in source
    assert "user_config = _resolved_user_config(per_cfg)" in source
    assert "OmegaConf.to_container(per_cfg, resolve=True)" not in source


def test_mixed_feature_selection_payload_remains_invalid() -> None:
    mixed = {**DEFAULT_FEATURE_SELECTION, **THRESHOLD_FEATURE_SELECTION}
    with pytest.raises(
        ValueError,
        match="cannot specify both 'distribution' and 'threshold_type'",
    ):
        parse_feature_selection(mixed)


def test_encdec_2xh200_run_script_uses_a_valid_threshold_config(
    hydra_compose,
) -> None:
    """No-GPU startup contract for the exact preset selected by the launcher."""
    script_path = REPO_ROOT / "runs" / "encdec_2xh200.sh"
    script = script_path.read_text()
    assert re.search(
        r"-m scripts\.train\.conditional_encdec_train\s*\\\n\s*--\s*\\\n",
        script,
    ), "torchrun launcher must separate its flags from script Hydra overrides"
    match = re.search(r"^\s*config_name=([A-Za-z0-9_]+)\s*\\?$", script, re.MULTILINE)
    assert match is not None, "run script must select a config_name=... preset"

    argv = [
        f"config_name={match.group(1)}",
        "feature_dir=out/data/features",
        "run=encdec-d26-full-2xh200",
    ]
    config_name, overrides = parse_config_name_and_overrides(
        argv,
        default_config_name="conditional_encdec_train",
    )
    cfg = hydra_compose(CONFIG_DIR, config_name, overrides=overrides)

    assert config_name == "conditional_encdec_2xh200"
    assert _as_dict(cfg.feature_selection) == THRESHOLD_FEATURE_SELECTION
    assert isinstance(parse_feature_selection(_as_dict(cfg.feature_selection)), ThresholdSelection)

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    "config_name",
    sorted(SHIPPED_SELECTION_CONFIGS),
)
def test_every_encoder_decoder_config_resolves_one_valid_selection_arm(
    hydra_compose,
    config_name: str,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    raw = _as_dict(cfg.feature_selection)
    parsed = parse_feature_selection(raw)
    assert parsed is not None


@pytest.mark.parametrize(
    ("config_name", "expected_sha256"),
    sorted(EXPECTED_RESOLVED_CONFIG_SHA256.items()),
)
def test_every_encoder_decoder_config_has_a_pinned_ordered_payload(
    hydra_compose,
    config_name: str,
    expected_sha256: str,
    _residual_head_artifact_environment: None,
) -> None:
    cfg = hydra_compose(CONFIG_DIR, config_name)
    payload = _as_dict(cfg)
    if config_name in SFT_CONFIGS | {RESIDUAL_HEAD_SFT_CONFIG, TRANSFORMER_HEAD_SFT_CONFIG}:
        # Preserve the historical payload oracle while checking the intentional
        # new fields separately. Do not repin unrelated pre-existing default drift.
        warmup = payload.pop("lr_warmup_steps")
        assert type(warmup) is int
        assert warmup == (0 if config_name == "conditional_encdec_sft_profile_4090" else 20)
        assert payload.pop("muon_parameter_packing") is False
    assert _ordered_json_sha256(payload) == expected_sha256


def test_shipped_selection_config_oracle_is_exhaustive() -> None:
    conditional_configs = {
        path.stem for path in (REPO_ROOT / "configs" / "training").glob("conditional_encdec*.yaml")
    }
    assert SHIPPED_SELECTION_CONFIGS == conditional_configs | {
        "bench_live_4090",
        "bench_prepacked_4090",
    }
    assert set(EXPECTED_RESOLVED_CONFIG_SHA256) == SHIPPED_SELECTION_CONFIGS


def test_conditional_encdec_docs_use_compose_api_config_selector() -> None:
    paths = [
        REPO_ROOT / "configs" / "training" / "conditional_encdec_4090.yaml",
        REPO_ROOT / "configs" / "training" / "conditional_encdec_2xb200.yaml",
        REPO_ROOT / "configs" / "training" / "conditional_encdec_2xh200.yaml",
        REPO_ROOT / "configs" / "training" / "conditional_encdec_4xh100.yaml",
        REPO_ROOT / "docs" / "reference" / "configuration.md",
        REPO_ROOT / "docs" / "pipelines" / "encoder-decoder-training.md",
    ]
    for path in paths:
        text = path.read_text()
        assert "--config-name conditional_encdec_" not in text, (
            f"{path.relative_to(REPO_ROOT)} documents Hydra's native selector "
            "for a compose-API entrypoint"
        )


def test_documented_sampling_defaults_match_the_composed_contract() -> None:
    pipeline_doc = (REPO_ROOT / "docs" / "pipelines" / "encoder-decoder-training.md").read_text()
    config_doc = (REPO_ROOT / "docs" / "reference" / "configuration.md").read_text()

    assert "| `feature_selection.mean` | 12.0 |" in pipeline_doc
    assert "| `feature_selection.std` | 12.0 |" in pipeline_doc
    assert "| `feature_selection.min_k` / `max_k` | 1 / 49 |" in pipeline_doc
    assert "| `feature_selection.delta_weights` | percentile_delta |" in pipeline_doc
    assert "The root config continues to use\n`prev_segment`" in pipeline_doc
    assert (
        "| `feature_selection` | geometric mean/std 12, `min_k=1`, "
        "`max_k=49`, `mode=top_k_percentile`, "
        "`delta_weights=percentile_delta` |"
    ) in config_doc
    assert "It does not itself enable counterfactual splicing." in config_doc


def test_cli_reference_quotes_zsh_sensitive_delete_override() -> None:
    text = (REPO_ROOT / "docs" / "cli-reference.md").read_text()
    assert "'~feature_selection'" in text
