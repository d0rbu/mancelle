"""Real-production-data regression tests for the activation-frequency
blacklist (issue #702). Gated behind `slow` + `requires_feature_data` + a
skip-if-missing check so this file is a no-op on any machine/CI without the
real gemma-3-4b-pt/l22/65k feature directory (~2GB+ of extraction output) --
skipping is not a failure -- and is excluded from the default `pytest` run
(`slow` is deselected by default addopts) since each test faithfully
reloads the real 82MB/65k-entry feature_statistics.safetensors from disk via
the actual production call signature (compute_activation_frequency_blacklist
takes a feature_dir and loads fresh internally, it does not accept a
pre-loaded stats object) -- the full file takes ~10 minutes on this machine,
confirmed by a real run during implementation. On a machine that has the
asset (this repo's canonical checkout), running it explicitly
(`-m slow` or by file path) exercises the exact production code path
against real data and pins the exact counts independently verified against
issue #702's table during planning -- turning a one-off manual verification
into a permanent, automatic regression gate.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from steering.conditional import freq_blacklist as fb
from steering.stats_io import find_statistics_path, load_statistics

_CANONICAL_ROOT = Path("/home/d0rb/Documents/Github/nano-gpt-pretrain-steer")
_ASSET_ROOT = Path(os.environ.get("ISSUE702_ASSET_ROOT", str(_CANONICAL_ROOT)))
_PROD_FEATURES = _ASSET_ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_N_SAE_FEATURES = 65536
_N_SEGMENTS = 130_534_655


def _skip_unless_prod_stats() -> Path:
    stats_path = _PROD_FEATURES / "feature_statistics.safetensors"
    if not stats_path.exists():
        pytest.skip(f"production feature_statistics.safetensors missing: {stats_path}")
    return _PROD_FEATURES


@pytest.mark.slow
@pytest.mark.requires_feature_data
def test_real_stats_n_segments_matches_recorded_corpus_size():
    """Pins the corpus size issue #702's table was measured against. If
    this fails, the on-disk statistics were regenerated from a different
    corpus and every count below must be re-derived, not blindly trusted."""
    feature_dir = _skip_unless_prod_stats()
    stats = load_statistics(find_statistics_path(feature_dir))  # ty: ignore[invalid-argument-type]
    assert int(stats.features.n_segments) == _N_SEGMENTS
    assert int(stats.metadata.n_sae_features) == _N_SAE_FEATURES


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.parametrize(
    "freq_min,expected_n_blacklisted",
    [
        (0.0, _N_SAE_FEATURES - 64642),
        (1e-6, _N_SAE_FEATURES - 63983),
        (1e-5, _N_SAE_FEATURES - 61835),
        (1e-4, _N_SAE_FEATURES - 51130),
        (1e-3, _N_SAE_FEATURES - 35275),
    ],
)
def test_real_stats_freq_min_sweep_matches_issue_702_table(
    freq_min, expected_n_blacklisted
):
    """Independently reproduces every freq_min row of issue #702's
    measurement table against the real production statistics, at the
    shipped automatic producer's freq_max=0.5. Verified by hand during planning; this test
    makes that verification permanent and automatic."""
    feature_dir = _skip_unless_prod_stats()
    res = fb.compute_activation_frequency_blacklist(
        feature_dir, _N_SAE_FEATURES, freq_min=freq_min, freq_max=0.5,
    )
    assert len(res.ids) == expected_n_blacklisted
    assert res.n_segments == _N_SEGMENTS


@pytest.mark.slow
@pytest.mark.requires_feature_data
def test_real_stats_new_default_blacklists_3701_features_at_the_default_freq_max():
    """The exact end-to-end number this plan's live verification produced
    (via the real precompute_feature_blacklist script, during planning):
    freq_min=1e-5, freq_max=0.5 (the automatic producer defaults after #702)
    blacklists exactly 3701 of 65536 features. The issue-#768 production run
    now uses an explicit audited whitelist, so this remains a producer
    regression rather than that run's model-shape contract."""
    feature_dir = _skip_unless_prod_stats()
    res = fb.compute_activation_frequency_blacklist(
        feature_dir, _N_SAE_FEATURES, freq_min=1e-5, freq_max=0.5,
    )
    assert len(res.ids) == 3701


@pytest.mark.slow
@pytest.mark.requires_feature_data
def test_real_stats_no_feature_left_with_fewer_than_300_activations_at_1e5():
    """The issue's core justification for 1e-5 specifically: it is the
    loosest freq_min that leaves zero surviving features with <300
    activations on this corpus. Pin that directly against real counts, and
    contrast with 1e-6 (looser), which must still leave exactly 481 such
    features -- proving 1e-5 is the actual transition point on this corpus,
    not an arbitrary choice."""
    feature_dir = _skip_unless_prod_stats()
    stats = load_statistics(find_statistics_path(feature_dir))  # ty: ignore[invalid-argument-type]
    counts = [0] * _N_SAE_FEATURES
    for fid_str, pfs in stats.per_feature_stats.items():
        counts[int(fid_str)] = int(pfs.count)

    res = fb.compute_activation_frequency_blacklist(
        feature_dir, _N_SAE_FEATURES, freq_min=1e-5, freq_max=0.5,
    )
    blacklisted = set(res.ids)
    survivors_under_300 = [
        fid for fid in range(_N_SAE_FEATURES)
        if fid not in blacklisted and 0 < counts[fid] < 300
    ]
    assert survivors_under_300 == []

    res_loose = fb.compute_activation_frequency_blacklist(
        feature_dir, _N_SAE_FEATURES, freq_min=1e-6, freq_max=0.5,
    )
    blacklisted_loose = set(res_loose.ids)
    survivors_loose_under_300 = [
        fid for fid in range(_N_SAE_FEATURES)
        if fid not in blacklisted_loose and 0 < counts[fid] < 300
    ]
    assert len(survivors_loose_under_300) == 481
