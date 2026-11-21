"""Full REAL-pipeline integration for gradient-attribution selection.

Requires pre-extracted real feature data (the ``requires_feature_data``
marker contract) with statistics, percentile overlays, token boundaries, and
packed rows. The SAE/model identity is read from env vars, defaulting to the
gemma-2-2b/16k/L8/s128/gemmascope config but adaptable to any real feature
set — e.g. gemma-3-4b-pt (which uses release=gemmascope_2 + an L0 tier):

    GRAD_E2E_MODEL=gemma-3-4b-pt GRAD_E2E_LAYER=... GRAD_E2E_SAE_RELEASE=gemmascope_2 \
    GRAD_E2E_SAE_WIDTH=... GRAD_E2E_SAE_L0=medium GRAD_E2E_SEGMENT_SIZE=... \
    GRAD_E2E_CONTEXT_WINDOW=... \
    uv run pytest tests/steering/test_gradient_attribution_e2e_real.py -m very_slow

When GRAD_E2E_FEATURE_DIR is unset it is derived from the params above
(``out/data/features/s{seg}/l{layer}/a_res/p_max_pool/f{width}/m_{model}/base``).

Covers, against REAL artifacts: the grad overlay producer (alignment
verification on real re-tokenization), packer-native provenance on real
packed rows, and gradient-ranked selection for the train AND val dirs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.requires_feature_data,
    pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA"),
]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# SAE / model identity — single source for the producer (step 1) and the
# selection consumer (step 3), so their overlay-descriptor checksums match.
_MODEL = _env("GRAD_E2E_MODEL", "gemma-2-2b")
_LAYER = int(_env("GRAD_E2E_LAYER", "8"))
_SAE_RELEASE = _env("GRAD_E2E_SAE_RELEASE", "gemmascope")
_SAE_WIDTH = _env("GRAD_E2E_SAE_WIDTH", "16k")
# "" => None (gemma-2 canonical). For gemma-3 gemmascope_2 set a tier, e.g.
# GRAD_E2E_SAE_L0=medium. Passed as a string; the producer coerces it.
_SAE_L0: str | None = _env("GRAD_E2E_SAE_L0", "") or None
_ACTIVATION_TYPE = _env("GRAD_E2E_ACTIVATION_TYPE", "residual")
_SEGMENT_SIZE = int(_env("GRAD_E2E_SEGMENT_SIZE", "128"))
_CONTEXT_WINDOW = int(_env("GRAD_E2E_CONTEXT_WINDOW", "2048"))
# Docs per padded forward/backward batch. Full-depth gradient attribution is
# VRAM-heavy; drop to 1 on a smaller card (gemma-3-4b at ctx 2048 OOMs a 24
# GiB GPU even at 1 — use a ~2b model there).
_BATCH_SIZE = int(_env("GRAD_E2E_BATCH_SIZE", "2"))
# Post-#533, extraction freezes BOS exclusion always-on: every new run writes a
# `no_bos` leaf, so that is the DEFAULT derived feature dir here. Set
# GRAD_E2E_LEGACY_BASE=1 for a legacy pre-freeze dir (bare `base`), or override
# GRAD_E2E_FEATURE_DIR for any other layout. The producer INHERITS the flag from
# the base shards regardless of which path this resolves to.
_LEGACY_BASE = _env("GRAD_E2E_LEGACY_BASE", "").strip().lower() in ("1", "true", "yes")
_BASE_LEAF = "base" if _LEGACY_BASE else "base/no_bos"

_FEATURE_DIR = Path(
    _env(
        "GRAD_E2E_FEATURE_DIR",
        f"out/data/features/s{_SEGMENT_SIZE}/l{_LAYER}/a_res/p_max_pool/"
        f"f{_SAE_WIDTH}/m_{_MODEL}/{_BASE_LEAF}",
    )
)
_GRAD_OVERRIDES = {
    "attribution_mode": "future_half_split",
    "min_final_segment_tokens": 8,
    "model_dtype": "bfloat16",
}


def _require(path: Path, hint: str) -> Path:
    if not path.is_dir():
        pytest.skip(f"missing {path} — {hint}")
    return path


@pytest.fixture(scope="module")
def real_dirs():
    feature_dir = _require(_FEATURE_DIR, "run the standard extraction pipeline first")
    pct_dirs = sorted((feature_dir / "pct").glob("*")) if (feature_dir / "pct").is_dir() else []
    if not pct_dirs:
        pytest.skip("no pct overlay — run precompute_percentiles")
    packed_candidates = sorted(feature_dir.glob("pct/*/tok/*/c*/sall"))
    if not packed_candidates:
        pytest.skip("no packed rows — run precompute_packed_rows")
    train_dir = packed_candidates[0]
    val_dir = train_dir.parent / f"val_{train_dir.name}"
    if not val_dir.is_dir():
        pytest.skip("no val packed rows")
    return {"feature_dir": feature_dir, "train_dir": train_dir, "val_dir": val_dir}


def test_step1_grad_overlay_producer(real_dirs):
    import scripts.preprocess.precompute_gradient_attribution as mod

    cfg: Any = OmegaConf.load(
        "configs/extraction/precompute_gradient_attribution.yaml"
    )
    cfg.feature_dir = str(real_dirs["feature_dir"])
    # SAE/model identity from env (defaults = the YAML's gemma-2-2b config).
    cfg.model = _MODEL
    cfg.layer = _LAYER
    cfg.activation_type = _ACTIVATION_TYPE
    cfg.segment_size = _SEGMENT_SIZE
    cfg.context_window = _CONTEXT_WINDOW
    cfg.sae_width = _SAE_WIDTH
    cfg.sae.release = _SAE_RELEASE
    cfg.sae.l0 = _SAE_L0
    # ignore_bos_features is INHERITED from the base shards by the producer — not
    # set here (a stale cfg key would only be logged + ignored).
    for k, v in _GRAD_OVERRIDES.items():
        cfg[k] = v
    cfg.resume = True
    cfg.batch_size = _BATCH_SIZE
    assert mod.run(cfg) == 0
    grad_dirs = list((real_dirs["feature_dir"] / "grad").iterdir())
    assert grad_dirs, "no grad overlay produced"


def test_step2_provenance_present(real_dirs):
    """Packed rows carry their segment provenance natively (written by the
    packer in the same pass). A dir packed before the integration heals via
    ``precompute_packed_rows resume=true``."""
    from steering.overlay_io import glob_overlay_shards, read_manifest

    for key in ("train_dir", "val_dir"):
        pdir = real_dirs[key]
        prov = pdir / "provenance"
        manifest = read_manifest(pdir)
        meta = (manifest or {}).get("metadata", {})
        assert prov.is_dir() and meta.get("base_shard_stems"), (
            f"{pdir} predates packer-integrated provenance — re-run "
            f"scripts.preprocess.precompute_packed_rows with resume=true"
        )
        assert len(glob_overlay_shards(prov)) == len(glob_overlay_shards(pdir))


def test_step3_gradient_selection_both_splits(real_dirs, tmp_path):
    from scripts.preprocess.precompute_selected_features import (
        ShardGradientJoin,
        resolve_gradient_artifacts,
        select_features_for_shard,
    )
    from steering.conditional.feature_pipeline import build_feature_pipeline
    from steering.conditional.sampling import parse_feature_selection
    from steering.overlay_io import glob_overlay_shards, load_overlay_shard
    from steering.packed_batch_io import load_selection_shard
    from steering.stats_io import find_statistics_path, load_statistics

    stats_path = find_statistics_path(real_dirs["feature_dir"])
    assert stats_path is not None
    stats = load_statistics(stats_path)
    fs_grad: Any = parse_feature_selection(
        {
            "distribution": "geometric",
            "mean": 5.0,
            "mode": "top_k_percentile",
            "min_k": 1,
            "max_k": 25,
            "gradient": {
                "mode": _GRAD_OVERRIDES["attribution_mode"],
                "model_dtype": _GRAD_OVERRIDES["model_dtype"],
                "min_final_segment_tokens": _GRAD_OVERRIDES["min_final_segment_tokens"],
            },
        }
    )
    sel_cfg = OmegaConf.create(
        {
            "sae": {
                "model": _MODEL, "layer": _LAYER, "release": _SAE_RELEASE,
                "width": _SAE_WIDTH, "l0": _SAE_L0, "activation_type": _ACTIVATION_TYPE,
            }
        }
    )
    for split in ("train_dir", "val_dir"):
        joiner, prov_dir, _checksum = resolve_gradient_artifacts(
            sel_cfg, real_dirs["feature_dir"], real_dirs[split], fs_grad.gradient
        )
        pipeline = build_feature_pipeline(
            feature_selection=fs_grad,
            negative_selection=None,
            per_feature_stats=stats.per_feature_stats,
            n_sae_features=stats.metadata.n_sae_features,
            num_buckets=2,
            seed=42,
        )
        out_dir = tmp_path / split
        out_dir.mkdir()
        shard_path = glob_overlay_shards(real_dirs[split])[0]
        join = ShardGradientJoin(joiner, load_overlay_shard(prov_dir / shard_path.name))
        select_features_for_shard(
            shard_path, pipeline, out_dir, shard_path.stem.split(".")[0],
            gradient_join=join,
        )
        sel = load_selection_shard(out_dir / shard_path.name)
        assert sel.total_selected > 0
        print(f"[{split}] selected {sel.total_selected} features across "
              f"{sel.total_segments} segments")
