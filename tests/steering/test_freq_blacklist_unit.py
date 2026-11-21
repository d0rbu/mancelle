"""Unit tests for the activation-frequency blacklist library (pure logic)."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from errors import FeatureMaskError
from steering.conditional import freq_blacklist as fb
from steering.conditional.feature_mask import _parse_ids_file as pr471_parse_ids
from tests.steering._freq_blacklist_fixtures import write_stats


class TestResolveBounds:
    def test_defaults_when_keys_absent(self):
        bounds = fb.resolve_blacklist_bounds(OmegaConf.create({}))
        assert bounds == fb.FrequencyBounds(min=1e-5, max=0.5)

    def test_both_explicit_null_is_disabled(self):
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": None, "feature_mask_freq_max": None}
        )
        assert fb.resolve_blacklist_bounds(cfg) is None  # both null ⇒ disabled

    def test_only_max_set_lb_skipped(self):
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": None, "feature_mask_freq_max": 0.2}
        )
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=None, max=0.2
        )

    def test_only_min_set_ub_skipped(self):
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": 0.1, "feature_mask_freq_max": None}
        )
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=0.1, max=None
        )

    def test_both_set(self):
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": 0.001, "feature_mask_freq_max": 0.5}
        )
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=0.001, max=0.5
        )

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan")])
    def test_freq_min_out_of_range_raises(self, bad):
        # freq_min must be in [0, 1) — 0 is OK (kill dead), 1 forbidden.
        # freq_max=None ISOLATES the range check: with the default freq_max a
        # wrongly-accepted bad freq_min would instead trip the fmin>=fmax
        # ordering error (whose message also names freq_min) → false green.
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": bad, "feature_mask_freq_max": None}
        )
        with pytest.raises(FeatureMaskError, match="feature_mask_freq_min"):
            fb.resolve_blacklist_bounds(cfg)

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.5, float("nan")])
    def test_freq_max_out_of_range_raises(self, bad):
        # freq_max must be in (0, 1] — 1 is OK (no UB filter), 0 forbidden.
        # freq_min=None isolates the range check from the ordering check.
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": None, "feature_mask_freq_max": bad}
        )
        with pytest.raises(FeatureMaskError, match="feature_mask_freq_max"):
            fb.resolve_blacklist_bounds(cfg)

    def test_freq_min_zero_accepted_alone(self):
        # The INCLUSIVE-lower edge of freq_min ([0, 1)) must be accepted —
        # 0.0 alone kills dead features. (Kills the lo_ok inclusive→exclusive
        # mutant, which the ordering check can't surface when max is None.)
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": 0.0, "feature_mask_freq_max": None}
        )
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=0.0, max=None
        )

    def test_freq_max_one_accepted_alone(self):
        # The INCLUSIVE-upper edge of freq_max ((0, 1]) must be accepted —
        # 1.0 disables the UB. (Kills the hi_ok inclusive→exclusive mutant.)
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": None, "feature_mask_freq_max": 1.0}
        )
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=None, max=1.0
        )

    @pytest.mark.parametrize("bad", [True, False, "x", [1]])
    def test_non_numeric_freq_min_raises(self, bad):
        cfg = OmegaConf.create({"feature_mask_freq_min": bad})
        with pytest.raises(FeatureMaskError, match="feature_mask_freq_min"):
            fb.resolve_blacklist_bounds(cfg)

    @pytest.mark.parametrize("bad", [True, False, "x", [1]])
    def test_non_numeric_freq_max_raises(self, bad):
        cfg = OmegaConf.create({"feature_mask_freq_max": bad})
        with pytest.raises(FeatureMaskError, match="feature_mask_freq_max"):
            fb.resolve_blacklist_bounds(cfg)

    @pytest.mark.parametrize(
        "lo,hi", [(0.5, 0.5), (0.6, 0.5), (0.99, 0.5)]
    )
    def test_min_must_be_strictly_less_than_max(self, lo, hi):
        cfg = OmegaConf.create(
            {"feature_mask_freq_min": lo, "feature_mask_freq_max": hi}
        )
        with pytest.raises(FeatureMaskError, match="must be strictly less"):
            fb.resolve_blacklist_bounds(cfg)

    def test_frequency_bounds_coerces_int_to_float(self):
        # __post_init__ must coerce int → float so downstream float math /
        # repr-based filenames are stable (1 and 1.0 must not differ on disk).
        b = fb.FrequencyBounds(min=0, max=1)
        assert type(b.min) is float and type(b.max) is float

    def test_unknown_legacy_key_is_ignored(self):
        # The old single-threshold key was never merged; it has no special
        # handling. A bare cfg carrying it just falls back to the defaults
        # (Hydra struct-mode rejects unknown CLI overrides at compose time).
        cfg = OmegaConf.create({"feature_mask_freq_threshold": 0.01})
        assert fb.resolve_blacklist_bounds(cfg) == fb.FrequencyBounds(
            min=1e-5, max=0.5
        )


class TestCanonicalBound:
    def test_bit_equal_floats_canonicalize_identically(self):
        assert fb.canonical_bound(0.01) == fb.canonical_bound(1e-2)
        assert fb.canonical_bound(0.01) == fb.canonical_bound(0.010)

    def test_distinct_thresholds_distinct(self):
        assert fb.canonical_bound(0.01) != fb.canonical_bound(0.02)

    def test_roundtrip(self):
        assert float(fb.canonical_bound(0.0123)) == 0.0123

    def test_filesystem_safe(self):
        for thr in (0.01, 1e-2, 0.5, 0.999, 1e-5, 0.0009999):
            tok = fb.canonical_bound(thr)
            assert all(c in "0123456789.e-" for c in tok), tok


class TestBlacklistPaths:
    def test_layout_and_bound_keying(self, tmp_path):
        txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        assert txt == tmp_path / "blacklist" / "freq_minnull_max0.01.txt"
        assert meta == (
            tmp_path / "blacklist" / "freq_minnull_max0.01.meta.json"
        )

    def test_both_bounds_in_filename(self, tmp_path):
        txt, _ = fb.blacklist_paths(tmp_path, freq_min=0.0, freq_max=0.99)
        assert txt == tmp_path / "blacklist" / "freq_min0.0_max0.99.txt"

    def test_only_min_set(self, tmp_path):
        txt, _ = fb.blacklist_paths(tmp_path, freq_min=0.001, freq_max=None)
        assert txt == tmp_path / "blacklist" / "freq_min0.001_maxnull.txt"

    def test_distinct_bounds_distinct_paths(self, tmp_path):
        a, _ = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        b, _ = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.02)
        assert a != b
        # Different min ⇒ different path even with same max
        c, _ = fb.blacklist_paths(tmp_path, freq_min=0.0, freq_max=0.01)
        assert c != a

    def test_bit_equal_bounds_same_path(self, tmp_path):
        a, _ = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        b, _ = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=1e-2)
        assert a == b


class TestComputeCore:
    def test_frequency_and_strict_boundary(self, tmp_path):
        # n_segments=100: f7 80/100=0.80, f42 25/100=0.25, f13 1/100=0.01
        write_stats(
            tmp_path, n_segments=100, counts={7: 80, 42: 25, 13: 1},
            n_sae_features=64,
        )
        res = fb.compute_activation_frequency_blacklist(tmp_path, 64, freq_min=None, freq_max=0.25)
        # f42 at EXACTLY 0.25 -> 25 > 0.25*100=25.0 is False -> excluded
        assert res.ids == (7,)
        assert res.n_segments == 100 and res.n_active_features == 3
        # one segment more for f42 -> included
        write_stats(
            tmp_path, n_segments=100, counts={7: 80, 42: 26, 13: 1},
            n_sae_features=64,
        )
        assert fb.compute_activation_frequency_blacklist(tmp_path, 64, freq_min=None, freq_max=0.25).ids == (7, 42)

    def test_inactive_features_never_blacklisted(self, tmp_path):
        # only f3 is active; f0,f1,f2 absent from per_feature_stats
        write_stats(tmp_path, n_segments=10, counts={3: 10}, n_sae_features=8)
        res = fb.compute_activation_frequency_blacklist(tmp_path, 8, freq_min=None, freq_max=0.5)
        assert res.ids == (3,)  # 10/10=1.0 > 0.5; absent features never appear

    def test_empty_result(self, tmp_path):
        write_stats(
            tmp_path, n_segments=100, counts={1: 1, 2: 2, 3: 3},
            n_sae_features=16,
        )
        res = fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.9)
        assert res.ids == () and res.is_empty

    def test_all_active_features_blacklisted(self, tmp_path):
        write_stats(
            tmp_path, n_segments=10, counts={0: 10, 1: 9, 2: 8},
            n_sae_features=4,
        )
        res = fb.compute_activation_frequency_blacklist(tmp_path, 4, freq_min=None, freq_max=0.001)
        assert res.ids == (0, 1, 2)

    def test_n_segments_zero_no_division(self, tmp_path):
        write_stats(tmp_path, n_segments=0, counts={1: 5}, n_sae_features=8)
        res = fb.compute_activation_frequency_blacklist(tmp_path, 8, freq_min=None, freq_max=0.01)
        assert res.n_segments == 0 and res.ids == ()

    def test_missing_stats_raises(self, tmp_path):
        with pytest.raises(FeatureMaskError, match="feature_statistics"):
            fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5)

    def test_range_guard_id_ge_n(self, tmp_path):
        # feature id 16 with n_sae_features=16 -> out of range [0,16)
        write_stats(tmp_path, n_segments=10, counts={16: 5}, n_sae_features=16)
        with pytest.raises(FeatureMaskError, match="out of range"):
            fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.1)

    def test_max_valid_id_accepted(self, tmp_path):
        write_stats(tmp_path, n_segments=10, counts={15: 9}, n_sae_features=16)
        assert 15 in fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5).ids


class TestBlacklistResult:
    def test_dedupes_and_sorts(self):
        r = fb.BlacklistResult(
            ids=(5, 1, 5, 2, 1),
            n_sae_features=16,
            n_segments=10,
            n_active_features=3,
            freq_min=None, freq_max=0.01,
            stats_generated_at="t",
        )
        assert r.ids == (1, 2, 5)
        assert not r.is_empty

    def test_empty(self):
        r = fb.BlacklistResult(
            ids=(),
            n_sae_features=16,
            n_segments=10,
            n_active_features=0,
            freq_min=None, freq_max=0.01,
            stats_generated_at="t",
        )
        assert r.is_empty


class TestPersistence:
    def _res(self, ids=(7, 12, 59)):
        return fb.BlacklistResult(
            ids=tuple(ids),
            n_sae_features=16384,
            n_segments=1000,
            n_active_features=900,
            freq_min=None, freq_max=0.01,
            stats_generated_at="2026-05-19T00:00:00+00:00",
        )

    def test_txt_parses_via_pr471(self, tmp_path):
        txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        fb.write_blacklist(txt, meta, self._res(ids=(3, 1, 2)))
        assert sorted(pr471_parse_ids(str(txt))) == [1, 2, 3]

    def test_txt_sorted_and_headered(self, tmp_path):
        txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        fb.write_blacklist(txt, meta, self._res(ids=(59, 7, 12)))
        lines = txt.read_text().splitlines()
        assert lines[0].startswith("#")
        ints = [int(x) for x in lines if x and not x.startswith("#")]
        assert ints == [7, 12, 59]

    def test_empty_result_written(self, tmp_path):
        txt, meta = fb.blacklist_paths(tmp_path, freq_min=None, freq_max=0.01)
        fb.write_blacklist(txt, meta, self._res(ids=()))
        assert '"empty": true' in meta.read_text()


# ---------------------------------------------------------------------------
# Two-bound semantic (freq_min lower-bound INCLUSIVE; freq_max upper EXCLUSIVE)
# ---------------------------------------------------------------------------


class TestLowerBound:
    def test_lb_zero_kills_dead_features_only(self, tmp_path):
        # n_sae=10; only fid 3 active (50/100=0.5). All other fids are dead
        # (count=0, absent from per_feature_stats).
        write_stats(
            tmp_path, n_segments=100, counts={3: 50}, n_sae_features=10
        )
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 10, freq_min=0.0, freq_max=None,
        )
        assert set(res.ids) == {0, 1, 2, 4, 5, 6, 7, 8, 9}
        assert 3 not in res.ids  # 0.5 > 0 (LB exclusive in KEEP direction)

    def test_lb_skipped_keeps_dead(self, tmp_path):
        # freq_min=None ⇒ no LB filtering; dead features stay un-blacklisted.
        write_stats(
            tmp_path, n_segments=100, counts={3: 50}, n_sae_features=10
        )
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 10, freq_min=None, freq_max=None,
        )
        assert res.ids == ()  # no LB, no UB ⇒ nothing blacklisted

    def test_lb_inclusive_boundary(self, tmp_path):
        # count=25, n_segments=100 ⇒ freq=0.25. With LB=0.25 (inclusive),
        # the feature is KILLED: 25 <= 0.25 * 100 = 25.0 is True.
        write_stats(
            tmp_path, n_segments=100, counts={5: 25, 7: 60},
            n_sae_features=10,
        )
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 10, freq_min=0.25, freq_max=None,
        )
        # 5 killed by LB (at boundary), 7 NOT killed (0.6 > 0.25),
        # and all dead features (0,1,2,3,4,6,8,9) killed by LB at any min≥0.
        assert 5 in res.ids
        assert 7 not in res.ids
        # +1 over the LB boundary survives
        write_stats(
            tmp_path, n_segments=100, counts={5: 26, 7: 60},
            n_sae_features=10,
        )
        res2 = fb.compute_activation_frequency_blacklist(
            tmp_path, 10, freq_min=0.25, freq_max=None,
        )
        assert 5 not in res2.ids
        assert 7 not in res2.ids


class TestBothBounds:
    def test_keep_open_below_closed_above(self, tmp_path):
        # n_sae=10, n_seg=100
        # fid 1: count=0  (dead)        → killed by LB (0 <= 0)
        # fid 2: count=1  (freq=0.01)   → kept (0 < 0.01 <= 0.99)
        # fid 3: count=50 (freq=0.5)    → kept
        # fid 4: count=99 (freq=0.99)   → kept (UB inclusive top: 99 > 99.0? No)
        # fid 5: count=100(freq=1.0)    → killed by UB (100 > 0.99*100=99)
        write_stats(
            tmp_path, n_segments=100,
            counts={2: 1, 3: 50, 4: 99, 5: 100},
            n_sae_features=10,
        )
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 10, freq_min=0.0, freq_max=0.99,
        )
        kept = set(range(10)) - set(res.ids)
        assert kept == {2, 3, 4}

    def test_record_carries_both_bounds(self, tmp_path):
        write_stats(
            tmp_path, n_segments=100, counts={1: 50}, n_sae_features=4
        )
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 4, freq_min=0.1, freq_max=0.99,
        )
        assert res.freq_min == 0.1 and res.freq_max == 0.99

    def test_n_segments_zero_blacklists_nothing(self, tmp_path):
        # No segments ⇒ frequency undefined ⇒ blacklist nothing, even with
        # LB set (we won't kill dead features when we have no statistics).
        write_stats(tmp_path, n_segments=0, counts={1: 5}, n_sae_features=8)
        res = fb.compute_activation_frequency_blacklist(
            tmp_path, 8, freq_min=0.0, freq_max=0.99,
        )
        assert res.ids == ()
