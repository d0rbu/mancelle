"""Tests for encoder / cross-attention / num_buckets recovery across a
deep checkpoint chain.

The encoder-decoder loader needs to locate the original encoder and
cross-attention configs for SFT checkpoints that don't store them
inline. Pretrain metas keep these under ``user_config.encoder`` /
``user_config.cross_attention``; SFT metas instead put them at the
top level as ``encoder_config`` / ``xattn_config``. And capabilities_sft
checkpoints are nested two directories below pretrain:

    <model_tag>/<prefix>/
      ├── meta_054600.json                       # pretrain: user_config.encoder
      └── safety_sft_conditioned_checkpoints/
          ├── meta_000886.json                   # safety_sft: top-level encoder_config
          └── capabilities_sft_conditioned_checkpoints/
              └── meta_001106.json               # capabilities_sft: top-level encoder_config

Before this fix, the recovery helper walked up exactly one parent
directory and required the inherited format to be identical, so a
capabilities_sft checkpoint would KeyError on ``user_config.encoder``
when its parent (safety_sft) only had a top-level ``encoder_config``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanochat.checkpoint_manager import _recover_encdec_ancestor_config


def _write_meta(
    path: Path,
    step: int,
    *,
    user_config: dict | None = None,
    top_level: dict | None = None,
) -> None:
    """Write a meta_<step>.json with optional user_config and top-level fields."""
    path.mkdir(parents=True, exist_ok=True)
    data: dict = {"step": step}
    if user_config is not None:
        data["user_config"] = user_config
    if top_level:
        data.update(top_level)
    meta_path = path / f"meta_{step:06d}.json"
    meta_path.write_text(json.dumps(data, indent=2))
    # Also create a stub model file so find_last_step picks up this step.
    (path / f"model_{step:06d}.pt").write_bytes(b"stub")


_PRETRAIN_ENCODER = {
    "n_embd": 0.5,
    "depth": 0.5,
    "n_head": None,
    "n_kv_head": None,
    "head_dim": 64,
}
_PRETRAIN_XATTN = {"frequency": 4, "n_head": None, "n_kv_head": None}
_NEGATIVE_SELECTION = {"distribution": "geometric", "mean": 2.0}


# ---------------------------------------------------------------------------
# Single-level recovery (legacy: works even before the fix)
# ---------------------------------------------------------------------------


class TestSingleLevelRecovery:
    def test_immediate_parent_has_user_config_encoder(self, tmp_path: Path):
        """SFT checkpoint one level below pretrain: parent has user_config.encoder."""
        pretrain = tmp_path / "d12" / "s128"
        _write_meta(
            pretrain, 54600,
            user_config={
                "encoder": _PRETRAIN_ENCODER,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )
        sft = pretrain / "safety_sft_checkpoints"
        _write_meta(sft, 886, user_config={"num_buckets": None})

        result = _recover_encdec_ancestor_config(str(sft))
        assert result["encoder"] == _PRETRAIN_ENCODER
        assert result["cross_attention"] == _PRETRAIN_XATTN
        assert result["num_buckets"] == 2


# ---------------------------------------------------------------------------
# The regression: capabilities_sft is two levels below pretrain
# ---------------------------------------------------------------------------


class TestDeepChainRecovery:
    def test_capabilities_sft_two_levels_below_pretrain(self, tmp_path: Path):
        """Walks past safety_sft (no encoder) up to pretrain (has it)."""
        pretrain = tmp_path / "d12" / "s128"
        _write_meta(
            pretrain, 54600,
            user_config={
                "encoder": _PRETRAIN_ENCODER,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 2,
                "negative_selection": _NEGATIVE_SELECTION,
            },
        )
        safety = pretrain / "safety_sft_conditioned_checkpoints"
        _write_meta(safety, 886, user_config={"num_buckets": None})
        caps = safety / "capabilities_sft_conditioned_checkpoints"
        _write_meta(caps, 1106, user_config={"num_buckets": None})

        result = _recover_encdec_ancestor_config(str(caps))
        assert result["encoder"] == _PRETRAIN_ENCODER
        assert result["cross_attention"] == _PRETRAIN_XATTN
        assert result["num_buckets"] == 2
        assert result["negative_selection"] == _NEGATIVE_SELECTION

    def test_parent_has_top_level_encoder_config_not_user_config(
        self, tmp_path: Path
    ):
        """Safety SFT meta carries top-level ``encoder_config`` (the observed
        d12_baseline format). Recovery should accept it.
        """
        pretrain = tmp_path / "d12" / "s128"
        _write_meta(
            pretrain, 54600,
            user_config={
                "encoder": _PRETRAIN_ENCODER,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )
        safety = pretrain / "safety_sft_conditioned_checkpoints"
        # Safety SFT: top-level encoder_config / xattn_config / num_buckets,
        # user_config has no encoder.
        _write_meta(
            safety, 886,
            user_config={"num_buckets": None},
            top_level={
                "encoder_config": _PRETRAIN_ENCODER,
                "xattn_config": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )
        caps = safety / "capabilities_sft_conditioned_checkpoints"
        _write_meta(
            caps, 1106,
            user_config={"num_buckets": None},
            top_level={
                "encoder_config": _PRETRAIN_ENCODER,
                "xattn_config": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )

        result = _recover_encdec_ancestor_config(str(caps))
        assert result["encoder"] == _PRETRAIN_ENCODER
        assert result["cross_attention"] == _PRETRAIN_XATTN
        assert result["num_buckets"] == 2

    def test_three_level_chain(self, tmp_path: Path):
        """Pretrain -> SFT_A -> SFT_B -> SFT_C. Walks all the way up."""
        root = tmp_path / "d12" / "s128"
        _write_meta(
            root, 54600,
            user_config={
                "encoder": _PRETRAIN_ENCODER,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )
        a = root / "a"; _write_meta(a, 100, user_config={"num_buckets": None})
        b = a / "b"; _write_meta(b, 50, user_config={"num_buckets": None})
        c = b / "c"; _write_meta(c, 5, user_config={"num_buckets": None})

        result = _recover_encdec_ancestor_config(str(c))
        assert result["encoder"] == _PRETRAIN_ENCODER


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNegativeCases:
    def test_raises_when_no_ancestor_has_encoder(self, tmp_path: Path):
        """A chain with no pretrain at the top should raise a clear error."""
        root = tmp_path / "lonely"
        _write_meta(root, 100, user_config={"num_buckets": None})
        leaf = root / "sub"
        _write_meta(leaf, 50, user_config={"num_buckets": None})
        with pytest.raises(FileNotFoundError, match="encoder"):
            _recover_encdec_ancestor_config(str(leaf))

    def test_raises_when_chain_has_no_metas(self, tmp_path: Path):
        """Walking above the filesystem's meta files still raises cleanly."""
        root = tmp_path / "empty_chain"
        leaf = root / "a" / "b"
        leaf.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="encoder"):
            _recover_encdec_ancestor_config(str(leaf))

    def test_num_buckets_from_ancestor_top_level(self, tmp_path: Path):
        """num_buckets can come from top-level when user_config.num_buckets is None."""
        root = tmp_path / "d"
        _write_meta(
            root, 100,
            user_config={
                "encoder": _PRETRAIN_ENCODER,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": None,
            },
            top_level={"num_buckets": 4},
        )
        leaf = root / "sft"
        _write_meta(leaf, 10, user_config={"num_buckets": None})
        result = _recover_encdec_ancestor_config(str(leaf))
        assert result["num_buckets"] == 4

    def test_prefers_closest_ancestor(self, tmp_path: Path):
        """When multiple ancestors have configs, the closest one wins."""
        root = tmp_path / "d"
        _write_meta(
            root, 100,
            user_config={
                "encoder": {"head_dim": 64, "n_embd": 0.5, "depth": 0.5,
                            "n_head": None, "n_kv_head": None},
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 2,
            },
        )
        mid = root / "safety"
        different_enc = {"head_dim": 128, "n_embd": 0.5, "depth": 0.5,
                         "n_head": None, "n_kv_head": None}
        _write_meta(
            mid, 10,
            user_config={
                "encoder": different_enc,
                "cross_attention": _PRETRAIN_XATTN,
                "num_buckets": 4,
            },
        )
        leaf = mid / "caps"
        _write_meta(leaf, 1, user_config={"num_buckets": None})

        result = _recover_encdec_ancestor_config(str(leaf))
        # Closest ancestor (`safety`) should win.
        assert result["encoder"]["head_dim"] == 128
        assert result["num_buckets"] == 4
