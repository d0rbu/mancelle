"""Review-driven hardening for the stats-file generator.

Pins the mutation-survivor-prone lines: the ``n_segments > 0`` guard, the
strict ``>``, BlacklistResult dedupe, the on-disk rewrite-correction logic,
the ${}-path guard, large-count float exactness, and the stats/config
n_sae_features-mismatch message.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_sft as sft_mod
import scripts.train.conditional_encdec_train as train_mod
from errors import FeatureMaskError
from steering._common.ddp import ddp_barrier
from steering.conditional import freq_blacklist as fb
from steering.conditional.feature_mask import resolve_feature_mask
from steering.stats_io import find_statistics_path, load_statistics, save_statistics
from tests.steering._freq_blacklist_fixtures import make_stats, write_stats


def test_n_segments_zero_guard_is_load_bearing(tmp_path):
    """count=5, n_segments=0, thr=0.5 → without the ``n_segments > 0`` guard
    ``5 > 0.5*0 == 0.0`` is True and the feature would be WRONGLY blacklisted.
    With the guard it is excluded. Kills the drop-guard mutant."""
    write_stats(tmp_path, n_segments=0, counts={5: 5}, n_sae_features=16)
    res = fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5)
    assert res.n_segments == 0
    assert res.ids == ()  # NOT (5,)


def test_strict_greater_not_geq(tmp_path):
    """count == thr*n_segments exactly must be EXCLUDED (kills `>`→`>=`)."""
    write_stats(tmp_path, n_segments=200, counts={4: 100}, n_sae_features=16)
    assert (
        fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5).ids
        == ()
    )  # 100 > 100.0 is False


def test_large_counts_float_exact(tmp_path):
    # int64 counts well below 2^53 → float64 promotion is exact at the boundary
    n = 1_000_000_003
    write_stats(
        tmp_path,
        n_segments=n,
        counts={1: n // 100, 2: n // 100 + 1},
        n_sae_features=16,
    )
    # thr = 0.01 ; cutoff = 0.01 * n ; f1.count = n//100 (≈ cutoff, below) ;
    # f2.count = n//100 + 1 (just above). Verify only f2 is blacklisted.
    res = fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.01)
    assert res.ids == (2,)


def test_blacklist_result_dedupes_and_sorts():
    r = fb.BlacklistResult(
        ids=(9, 1, 9, 3, 1),
        n_sae_features=16,
        n_segments=10,
        n_active_features=3,
        freq_min=None,
        freq_max=0.01,
        stats_generated_at="t",
    )
    assert r.ids == (1, 3, 9)


def test_externally_corrupted_txt_is_corrected(tmp_path):
    """rank-0 recomputes from stats every call; if the on-disk id set differs
    (e.g. hand-edited / stale), it is rewritten — never trusted blindly."""
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    txt_path, meta_path = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    txt_path.write_text("# tampered\n2\n7\n")  # wrong ids
    out = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out is not None and out[1].ids == (1,)  # corrected from stats
    assert sorted(fb._parse_ids_file(txt_path)) == [1]


def test_meta_missing_forces_rewrite_even_if_txt_matches(tmp_path):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    _txt, meta_path = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    meta_path.unlink()
    out = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)
    assert out is not None and meta_path.is_file()  # meta regenerated


def test_dollar_brace_path_rejected(tmp_path, monkeypatch):
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    real = fb.generate_or_load_blacklist

    def fake(*a, **k):
        out = real(*a, **k)
        if out is None:
            return None
        _txt, res = out
        return Path("/tmp/${oc.env:HOME}/blacklist.txt"), res

    monkeypatch.setattr(fb, "generate_or_load_blacklist", fake)
    with pytest.raises(FeatureMaskError, match=r"interpolation"):
        fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)


def test_stats_width_mismatch_message(tmp_path):
    """stats metadata n_sae_features=16 but configured n_sae_features=8 and a
    feature id 12 (valid in 16, invalid in 8) → actionable mismatch error."""
    write_stats(tmp_path, n_segments=10, counts={12: 5}, n_sae_features=16)
    with pytest.raises(FeatureMaskError, match="disagrees with the configured"):
        fb.compute_activation_frequency_blacklist(tmp_path, 8, freq_min=None, freq_max=0.1)


def test_applied_then_resolve_mask_excludes(tmp_path):
    """End-to-end: generated file → PR 471 FeatureMask excludes exactly the
    high-frequency feature."""
    write_stats(tmp_path, n_segments=10, counts={1: 9, 2: 1}, n_sae_features=16)
    cfg = OmegaConf.create(
        {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4, "feature_mask": None}
    )
    fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    mask = resolve_feature_mask(cfg, 16)
    assert mask is not None and mask.ids == (1,)


# --- Review-2: kill surviving mutations / close confidence gaps -------------


def test_bool_bound_rejected_as_type_not_range():
    """E7: True/False must be rejected by the TYPE branch independently of the
    range check — assert the message names the type (the range branch never
    does), so the bool guard is killed on its own merits."""
    for bad in (True, False):
        cfg = OmegaConf.create({"feature_mask_freq_max": bad})
        with pytest.raises(
            FeatureMaskError,
            match=r"got bool",
        ):
            fb.resolve_blacklist_bounds(cfg)


def test_negative_feature_id_rejected(tmp_path):
    """M6b: the ``fid < 0`` half of the range guard is load-bearing."""
    write_stats(tmp_path, n_segments=10, counts={-1: 5, 3: 9}, n_sae_features=16)
    with pytest.raises(FeatureMaskError, match="out of range"):
        fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.1)


def test_applies_when_feature_mask_key_absent(tmp_path):
    """E5b: a cfg WITHOUT a pre-declared feature_mask key relies on
    OmegaConf.update(force_add=True). Pin that it is load-bearing."""
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    cfg = OmegaConf.create({"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4})  # no key
    out = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    assert out.status == fb.FrequencyBlacklistStatus.APPLIED
    assert cfg.feature_mask.mode == "blacklist"


def test_force_add_load_bearing_under_struct_mode(tmp_path):
    """E5b done right: real Hydra configs are STRUCT-mode, where adding the
    missing `feature_mask` key requires force_add=True (a plain non-struct
    OmegaConf.create would accept the add regardless — false green). Pins that
    force_add is genuinely load-bearing on the production-shaped config."""
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    cfg = OmegaConf.create({"feature_mask_freq_min": None, "feature_mask_freq_max": 0.4})
    OmegaConf.set_struct(cfg, True)  # the bit the plain test omits
    out = fb.maybe_apply_frequency_blacklist(cfg, str(tmp_path), 16)
    assert out.status == fb.FrequencyBlacklistStatus.APPLIED
    assert cfg.feature_mask.mode == "blacklist"


def test_fixture_stats_roundtrip_exact(tmp_path):
    """The load-bearing serde assumption: save_statistics → load_statistics
    preserves per-feature ``count`` and ``n_segments`` EXACTLY (and the
    count-descending order) for the synthetic fixture itself."""
    counts = {7: 80, 42: 25, 13: 1, 0: 999_999}
    write_stats(tmp_path, n_segments=123_457, counts=counts, n_sae_features=64)
    st = load_statistics(find_statistics_path(tmp_path))  # ty: ignore[invalid-argument-type]
    assert int(st.features.n_segments) == 123_457
    assert {int(k): v.count for k, v in st.per_feature_stats.items()} == counts
    assert [v.count for v in st.per_feature_stats.values()] == sorted(counts.values(), reverse=True)


def test_uses_exact_count_not_rounded_pct(tmp_path):
    """M3 by-design: count/n_segments is exact at the boundary even when a
    realistic (rounded) pct_segments_active would cross it. count=4999,
    n_segments=10000, thr=0.5 → cutoff 5000.0 → 4999 > 5000 is False
    (EXCLUDED). A pct of 0.5 (round(0.4999,4)) would WRONGLY include."""
    s = make_stats(n_segments=10000, counts={3: 4999}, n_sae_features=16)
    object.__setattr__(s.per_feature_stats["3"], "pct_segments_active", 0.5)
    save_statistics(s, tmp_path / "feature_statistics.safetensors")
    assert (
        fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5).ids
        == ()
    )


def test_two_generations_byte_identical_txt(tmp_path):
    """Persisted-file determinism: two independent generations of the same
    (stats, threshold) yield byte-identical blacklist.txt."""
    write_stats(tmp_path, n_segments=10, counts={1: 8, 2: 1}, n_sae_features=16)
    p1 = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)[0].read_bytes()  # ty: ignore[not-subscriptable]
    txt_path, meta_path = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.4)
    txt_path.unlink()
    meta_path.unlink()
    p2 = fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)[0].read_bytes()  # ty: ignore[not-subscriptable]
    assert p1 == p2


def test_ddp_rank0_error_other_ranks_still_barrier(tmp_path, monkeypatch):
    """M1 (NCCL-deadlock) pinned on the NON-success path too: when rank 0
    would fail, a rank>0 caller must still reach the barrier (it never
    computes — the artifact is read from a prior write)."""
    write_stats(tmp_path, n_segments=10, counts={1: 8}, n_sae_features=16)
    fb.generate_or_load_blacklist(tmp_path, 16, freq_min=None, freq_max=0.4)  # seed the artifact
    n = {"b": 0}

    def barrier():
        n["b"] += 1

    def _explode(*a, **k):
        raise AssertionError("rank>0 must not compute")

    monkeypatch.setattr(fb, "compute_activation_frequency_blacklist", _explode)
    fb.generate_or_load_blacklist(
        tmp_path, 16, freq_min=None, freq_max=0.4, ddp_rank=1, barrier_fn=barrier
    )
    assert n["b"] == 1  # rank>0 reached the barrier (0 if barrier mis-scoped)


def test_canonical_bound_submillesimal_exact():
    """E11: bit-exact path keying for small-but-realistic thresholds."""
    for thr in (0.0001, 0.00025, 0.0005, 0.0009999, 1e-5):
        assert float(fb.canonical_bound(thr)) == thr
        assert fb.canonical_bound(thr) != fb.canonical_bound(thr * 2)


def test_width_equality_mismatch_even_when_all_ids_in_range(tmp_path):
    """Review-1 MAJOR #1: a stats file for a DIFFERENT SAE width must
    fail-fast EVEN when every active id is < the configured width (the
    per-feature range guard alone would not catch this → silent-wrong)."""
    # stats width 64, all active ids < 32, configured width 32
    write_stats(tmp_path, n_segments=10, counts={1: 9, 2: 8, 3: 7}, n_sae_features=64)
    with pytest.raises(FeatureMaskError, match="disagrees with the configured"):
        fb.compute_activation_frequency_blacklist(tmp_path, 32, freq_min=None, freq_max=0.5)


def test_pretrain_wiring_stays_structural_while_sft_drops_the_producer():
    """Pretrain retains both producer sites; SFT inherits an explicit universe."""
    bss = inspect.getsource(train_mod.build_shared_state)
    toc = inspect.getsource(train_mod.train_one_config)
    main = inspect.getsource(sft_mod.main)
    # called in BOTH train functions, BEFORE a resolve_feature_mask there
    for body in (bss, toc):
        assert "maybe_apply_frequency_blacklist(" in body
        i = body.index("maybe_apply_frequency_blacklist(")
        j = body.index("resolve_feature_mask(")
        assert i < j, "maybe_apply must precede resolve_feature_mask"
        assert "barrier_fn=ddp_barrier" in body
    assert "maybe_apply_frequency_blacklist(" not in main
    assert "barrier_fn=ddp_barrier" not in main
    assert "resolve_feature_mask(" in main
    assert train_mod.ddp_barrier is ddp_barrier
    assert not hasattr(sft_mod, "ddp_barrier")


def test_metric_is_per_segment_not_per_token(tmp_path):
    """SEMANTIC PIN — the threshold is a fraction of post-pooling SEGMENTS,
    not tokens. The pipeline pools `segment_size` tokens per segment before
    SAE encoding (extract_features.py:pool_one_segment), so the only
    granularity persisted in feature_statistics.safetensors is per-segment
    count. This fixture DISCRIMINATES the two interpretations:

      count=50, n_segments=100, segment_size=128, freq_min=None, freq_max=0.01
        per-segment freq    = 50 / 100         = 0.50   > 0.01 → BLACKLIST
        per-token-equiv     = 50 / (100*128)   ≈ 0.0039 < 0.01 → would NOT
                              ( = count / n_tokens_gpt2 )

    The implementation MUST choose per-segment. If a future refactor flips
    the metric to per-token this test will fail loudly.
    """
    write_stats(tmp_path, n_segments=100, counts={3: 50}, n_sae_features=8)
    res = fb.compute_activation_frequency_blacklist(tmp_path, 8, freq_min=None, freq_max=0.01)
    # Per-segment interpretation: 50 > 0.01*100 = 1.0 → blacklisted.
    # (A per-token interpretation would give 50/(100*128) ≈ 0.0039 < 0.01.)
    assert res.ids == (3,), "metric drifted to per-token (or worse); freq must be count/n_segments"
    assert res.n_segments == 100  # the denominator we used
