"""Integration: cfg mutation, PR 471 interop, training-script wiring."""

from __future__ import annotations

from pathlib import Path

import torch as t
import yaml
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_sft as sft_mod
import scripts.train.conditional_encdec_train as train_mod
import steering._common.ddp as ddp_mod
from steering._common.ddp import ddp_barrier
from steering.conditional import freq_blacklist as fb
from steering.conditional.feature_mask import (
    canonical_dict,
    feature_mask_dirname,
    feature_mask_leaf_from_cfg,
    resolve_feature_mask,
)
from steering.packed_batch_io import compute_selection_hash
from tests.steering._freq_blacklist_fixtures import write_stats

# Anchored to the REPO, not the CWD. A CWD-relative "configs/training/..." reads whatever
# happens to sit under the directory pytest was invoked from — so the lockstep assertions
# below would silently check the wrong file, or vanish into a FileNotFoundError, depending on
# where the suite was launched. Every sibling test module anchors the same way.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "training"

POS = {"threshold_type": "segment_topk", "threshold": 2048}


def test_applied_sets_cfg_and_resolves(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16384)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    outcome = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert outcome.status == fb.FrequencyBlacklistStatus.APPLIED
    assert cfg.feature_mask.mode == "blacklist"
    mask = resolve_feature_mask(cfg, 16384)
    assert mask is not None and mask.ids == (1,)


def test_disabled_leaves_cfg_untouched(tmp_path):
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": None, "feature_mask": None}
    )
    outcome = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert outcome.status == fb.FrequencyBlacklistStatus.DISABLED
    assert cfg.feature_mask is None
    assert canonical_dict(resolve_feature_mask(cfg, 16384)) is None


def test_empty_noop_leaves_cfg_untouched(tmp_path):
    write_stats(tmp_path, n_segments=100, counts={1: 1, 2: 2}, n_sae_features=16)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.9, "feature_mask": None}
    )
    outcome = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    assert outcome.status == fb.FrequencyBlacklistStatus.EMPTY_NOOP
    assert cfg.feature_mask is None


def test_user_mask_precedence(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16384)
    cfg = OmegaConf.create(
        {
            "feature_mask_freq_min": None,
            "feature_mask_freq_max": 0.4,
            "feature_mask": {"mode": "blacklist", "ids": [99]},
        }
    )
    outcome = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert outcome.status == fb.FrequencyBlacklistStatus.SKIPPED_USER_MASK
    assert list(cfg.feature_mask.ids) == [99]


def test_missing_feature_dir_skips_gracefully():
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    outcome = fb.maybe_apply_frequency_blacklist(cfg, None, 16384)
    assert outcome.status == fb.FrequencyBlacklistStatus.SKIPPED_NO_FEATURE_DIR
    assert cfg.feature_mask is None


def test_idempotent_second_call(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16384)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    out2 = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16384)
    assert out2.status == fb.FrequencyBlacklistStatus.SKIPPED_USER_MASK


def test_end_to_end_pr471_excludes_blacklisted(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    mask = resolve_feature_mask(cfg, 16)
    assert mask is not None and mask.mode == "blacklist" and mask.ids == (1,)
    keep = mask.keep_mask(t.tensor([0, 1, 2], dtype=t.int64))
    assert keep.tolist() == [True, False, True]
    assert bool(mask.allowed_universe_mask(16)[1].item()) is False


def test_generated_path_dirname_stable(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    leaf = feature_mask_dirname(resolve_feature_mask(cfg, 16))
    # PR 471 dirname = "bl_" + sha256-hex prefix (length is PR 471's choice;
    # don't hardcode it — assert the structural + consistency invariant).
    assert leaf.startswith("bl_") and len(leaf) > 3
    assert all(c in "0123456789abcdef" for c in leaf[3:])
    # The two PR 471 entry points must agree on the leaf for our mutated cfg.
    assert feature_mask_leaf_from_cfg(cfg) == leaf


def test_fast_sweep_per_cfg_consistent_mask(tmp_path):
    """B1: build_shared_state(base_cfg) then train_one_config(fresh per_cfg)
    must resolve to the SAME mask (no masked-discovery / unmasked-training
    split-brain in run_encdec_sweep_fast)."""
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    base_cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    o1 = fb.maybe_apply_frequency_blacklist(base_cfg, str(tmp_path), 16)
    assert o1.status == fb.FrequencyBlacklistStatus.APPLIED
    base_hash = compute_selection_hash(
        POS,
        None,
        None,
        10,
        42,
        0,
        feature_mask=canonical_dict(resolve_feature_mask(base_cfg, 16)),
    )
    per_cfg = OmegaConf.create(  # fresh compose; mutation NOT inherited
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    o2 = fb.maybe_apply_frequency_blacklist(per_cfg, str(tmp_path), 16)
    assert o2.status == fb.FrequencyBlacklistStatus.APPLIED
    per_hash = compute_selection_hash(
        POS,
        None,
        None,
        10,
        42,
        0,
        feature_mask=canonical_dict(resolve_feature_mask(per_cfg, 16)),
    )
    assert base_hash == per_hash
    base_mask = resolve_feature_mask(base_cfg, 16)
    per_mask = resolve_feature_mask(per_cfg, 16)
    assert base_mask is not None and per_mask is not None
    assert base_mask.ids == per_mask.ids == (1,)


def test_train_script_wires_producer_and_shared_barrier():
    # Both the entry point and the shared DDP barrier are imported (the local
    # per-script barrier was de-duplicated into steering._common.ddp).
    assert hasattr(train_mod, "maybe_apply_frequency_blacklist")
    assert train_mod.ddp_barrier is ddp_barrier


def test_sft_script_drops_producer_and_its_unused_barrier_import():
    assert not hasattr(sft_mod, "maybe_apply_frequency_blacklist")
    assert not hasattr(sft_mod, "ddp_barrier")


def test_ddp_barrier_is_noop_without_dist():
    # The shared barrier no-ops in single-process / CPU runs (DDP not init).
    ddp_barrier()  # must not raise


def test_ddp_barrier_calls_dist_when_initialized(monkeypatch):
    # When DDP IS initialized, the shared barrier delegates to dist.barrier().
    calls = {"n": 0}
    monkeypatch.setattr(ddp_mod, "is_ddp_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "barrier", lambda: calls.__setitem__("n", calls["n"] + 1))
    ddp_mod.ddp_barrier()
    assert calls["n"] == 1


def test_pretrain_yaml_has_threshold_default_but_sft_drops_the_producer_keys():
    """Pretrain remains in lockstep while SFT inherits an explicit base universe.

    The band is ``(1e-5, 0.5)``: kill dead features and those too rare for the autointerp
    judge to score (``freq_min``, #702), and those active in more than half of all segments
    (``freq_max``).

    ``freq_max`` is 0.5, not #702's shipped 0.99, on the owner's call — and it reconciles
    #702 with itself: every row of #702's own measurement table is computed at
    ``freq_max=0.5`` (see ``test_freq_blacklist_real_data.py``, and the queued run pins 0.5),
    so the 0.99 constant never matched the analysis that justified it.

    The band is a SELECTION policy, and #695 does not get a vote: an earlier draft of #695
    raised it to ``(0.0001, 0.5)`` to make compaction "worth it", but compaction's runtime
    cost is a single gather, so it pays at ANY band and has no business narrowing the
    research universe to flatter its own numbers.
    """
    p = CONFIGS / "conditional_encdec_train.yaml"
    data = yaml.safe_load(p.read_text())
    # Lockstep, not just value-equality: the module constants are only a FALLBACK for an
    # ABSENT key, so the YAML wins at runtime. A drifted pair would silently give the
    # producer and a default training run DIFFERENT blacklists — and since #695 that is a
    # different ENCODER SHAPE, not merely a different id list.
    assert data["feature_mask_freq_min"] == fb.DEFAULT_FREQ_MIN == 1e-5
    assert data["feature_mask_freq_max"] == fb.DEFAULT_FREQ_MAX == 0.5
    assert data["feature_mask"] is None

    sft_data = yaml.safe_load((CONFIGS / "conditional_encdec_sft.yaml").read_text())
    assert "feature_mask_freq_min" not in sft_data
    assert "feature_mask_freq_max" not in sft_data
    assert sft_data["feature_mask"] is None


def test_train_yaml_has_the_fp_head_OFF_by_default():
    """#695: feature_prediction is a THIRD feature-id-indexed table (50,397,184 params at
    the 65k/768 shape) and is not yet compaction-aware, so enabling it alongside a
    feature_mask raises NotImplementedError. It is therefore null by default."""
    data = yaml.safe_load((CONFIGS / "conditional_encdec_train.yaml").read_text())
    assert data["feature_prediction"] is None
