"""Unit tests pinning axbench_bpb_eval.yaml defaults.

A mix of axbench-base parity (disabled_methods, num_buckets, LoRA
rank/alpha — the latter pinned by f24f1969 to match the rank of saved
per-concept adapters) and intentional BPB-only divergences
(diffmean_and_probe_layer=8 vs axbench-base 6 — the BPB d12 encdec
hook layer). The tests below pin both the parity AND the divergence
so a future drift in either direction is caught.
"""

from __future__ import annotations

import os

from hydra import compose, initialize_config_dir

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.eval.axbench_bpb.config import AxBenchBpbConfig, ContextCacheConfig


def _compose_default_bpb_cfg():
    """Compose the BPB YAML the same way the CLI does, but in-process."""
    # Find repo root (this file is at tests/axbench_bpb/unit/test_axbench_bpb_yaml_defaults.py).
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    cfg_dir = os.path.join(repo_root, "configs", "eval", "axbench_bpb")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="axbench_bpb_eval")


def test_diffmean_and_probe_layer_default_is_eight():
    """BPB pins diffmean/probe layer at 8 — distinct from axbench-base's
    LayerHooksConfig.diffmean_and_probe_layer=6 default. The BPB YAML
    set 8 to align with the d12 encdec residual hook layer; the
    docstring's earlier claim that "BPB defaults match axbench-base"
    was a Phase-W8 design that drifted as BPB matured. Pin the
    intentional divergence so a future revert is caught."""
    cfg = _compose_default_bpb_cfg()
    assert cfg.diffmean_and_probe_layer == 8


def test_lora_rank_default_is_eight():
    """BPB pins LoRA rank at 8 to match the rank of saved per-concept
    LoRA adapters under ``<source_run_dir>/baselines/lora/``. axbench-base's
    LoraConfig defaults to rank=8, so the saved ``lora_A.weight`` has
    shape ``(rank=8, d_in)``. A BPB default of 16 (the prior value)
    tripped a shape mismatch inside ``apply_intervention`` during
    Stage 3 (observed 2026-05-13 on the d12_s32 run); see commit
    f24f1969 for the full diagnosis. Override on the CLI if scoring a
    run dir whose adapters were trained with a different rank.
    """
    cfg = _compose_default_bpb_cfg()
    assert cfg.lora.rank == 8


def test_lora_alpha_default_is_sixteen():
    """BPB pins LoRA alpha at 16 to match axbench-base's LoraConfig
    default. Paired with rank=8 above per commit f24f1969 to preserve
    the alpha/rank=2 ratio used at adapter-training time."""
    cfg = _compose_default_bpb_cfg()
    assert cfg.lora.alpha == 16.0


def test_disabled_methods_default_keeps_fair_feature_diffmean_enabled():
    cfg = _compose_default_bpb_cfg()
    disabled = list(cfg.disabled_methods or [])
    assert "ours_gcg_encdec" in disabled
    assert "neg_ours_gcg_encdec" in disabled
    assert "ours_gcg_roofline_encdec" in disabled
    assert "neg_ours_gcg_roofline_encdec" in disabled
    assert "feature_diffmean_encdec" not in disabled
    assert "neg_feature_diffmean_encdec" not in disabled
    assert "feature_diffmean_roofline_encdec" in disabled
    assert "neg_feature_diffmean_roofline_encdec" in disabled


def test_feature_diffmean_top_k_defaults_resolve_from_checkpoint_metadata():
    cfg = _compose_default_bpb_cfg()
    assert cfg.feature_diffmean.top_k_pos is None
    assert cfg.feature_diffmean.top_k_neg is None


def test_filter_methods_applied_with_default_disabled_drops_bpb_adv_only_methods():
    """End-to-end on the method registry: filter_methods with the
    default disabled list must NOT include BPB-Adv-only precomputed methods."""
    from steering.eval.axbench_base.method_registry import (
        ALL_METHODS,
        filter_methods,
    )

    cfg = _compose_default_bpb_cfg()
    disabled = list(cfg.disabled_methods or [])
    filtered = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=disabled,
        target_model_has_negative_sampling=False,
    )
    names = [m.name for m in filtered]
    assert "ours_gcg_encdec" not in names
    assert "neg_ours_gcg_encdec" not in names
    assert "ours_gcg_roofline_encdec" not in names
    assert "neg_ours_gcg_roofline_encdec" not in names
    assert "feature_diffmean_encdec" in names
    assert "neg_feature_diffmean_encdec" in names
    assert "feature_diffmean_roofline_encdec" not in names
    assert "neg_feature_diffmean_roofline_encdec" not in names


def test_context_cache_num_buckets_default_is_2_in_dataclass():
    """Pin the ContextCacheConfig.num_buckets default at 2.

    Commit 0423c939 changed this from 10 to 2 because the d12_s32 /
    d12_baseline encdec checkpoints in this codebase have
    encoder.magnitude_embed.weight of shape (2, 384). A default >= 3
    would index out of bounds in vectorized_gather_kernel on CUDA.
    """
    assert ContextCacheConfig().num_buckets == 2, (
        "ContextCacheConfig.num_buckets default is no longer 2; see "
        "commit 0423c939 for the reason (encdec encoder magnitude "
        "embedding shape)."
    )


def test_extraction_max_tokens_default_is_default_context_window_in_dataclass():
    """Pin ``AxBenchBpbConfig.extraction_max_tokens`` at
    ``DEFAULT_CONTEXT_WINDOW`` (1024), NOT a literal 2048.

    Commit 756e401e ("standardize pipeline context window at 1024")
    introduced ``steering.context_defaults.DEFAULT_CONTEXT_WINDOW`` on
    main AFTER the PR #739 Track F oracle branch forked; the oracle's
    own ``config.py`` still carries a literal ``extraction_max_tokens:
    int = 2048``. A literal port of that line would silently regress
    the effective default from 1024 to 2048 for every unclustered
    caller -- a regression invisible from the oracle side (the oracle's
    value legitimately IS 2048 there). See
    docs/plans/2026-07-24-pr739-track-f-plan-part1.md MAIN-DRIFT
    CHECK #3.
    """
    assert DEFAULT_CONTEXT_WINDOW == 1024, (
        "DEFAULT_CONTEXT_WINDOW itself drifted off 1024; re-derive the "
        "expected extraction_max_tokens default before trusting the "
        "assertion below."
    )
    assert AxBenchBpbConfig().extraction_max_tokens == 1024
    assert AxBenchBpbConfig().extraction_max_tokens == DEFAULT_CONTEXT_WINDOW


def test_extraction_max_tokens_default_is_1024_in_yaml():
    """The Hydra-composed default also resolves to 1024, matching the
    dataclass default above. Guards the same MAIN-DRIFT CHECK #3
    regression at the YAML layer -- a future edit to
    configs/eval/axbench_bpb/axbench_bpb_eval.yaml that reintroduces a
    literal 2048 (mirroring the oracle's YAML) would be caught here."""
    cfg = _compose_default_bpb_cfg()
    assert cfg.extraction_max_tokens == 1024


def test_context_cache_num_buckets_default_is_2_in_yaml():
    """The Hydra-composed default also matches — caught a regression
    where the YAML and dataclass defaults could drift."""
    cfg = _compose_default_bpb_cfg()
    assert cfg.context_cache.num_buckets == 2, (
        f"axbench_bpb_eval.yaml's context_cache.num_buckets is not 2; "
        f"got {cfg.context_cache.num_buckets}. See commit 0423c939."
    )
