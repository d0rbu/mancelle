"""Property-based correctness for the freq-blacklist core (hypothesis)."""
from __future__ import annotations

import random

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional import freq_blacklist as fb
from tests.steering._freq_blacklist_fixtures import write_stats


@st.composite
def _corpus(draw):
    n_sae = draw(st.integers(8, 128))
    n_segments = draw(st.integers(0, 5000))
    n_active = draw(st.integers(0, min(n_sae, 40)))
    fids = draw(
        st.lists(
            st.integers(0, n_sae - 1),
            min_size=n_active,
            max_size=n_active,
            unique=True,
        )
    )
    counts = {}
    for fid in fids:
        # active features always have count >= 1; cap at n_segments (a feature
        # cannot fire in more segments than exist) or 1 when n_segments == 0
        hi = max(1, n_segments)
        counts[fid] = draw(st.integers(1, hi))
    thr = draw(st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
    return n_sae, n_segments, counts, thr


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=_corpus())
def test_matches_independent_oracle(tmp_path_factory, data):
    n_sae, n_segments, counts, thr = data
    d = tmp_path_factory.mktemp("c")
    write_stats(d, n_segments=n_segments, counts=counts, n_sae_features=n_sae)
    # Independent oracle: same canonical predicate, computed directly from the
    # input counts dict (not via the stats round-trip / library iteration).
    if n_segments <= 0:
        oracle: tuple[int, ...] = ()
    else:
        cutoff = float(thr) * float(n_segments)
        oracle = tuple(
            sorted(f for f, c in counts.items() if float(c) > cutoff)
        )
    res = fb.compute_activation_frequency_blacklist(
        d, n_sae, freq_min=None, freq_max=thr
    )
    assert res.ids == oracle
    # determinism: identical second derivation
    assert fb.compute_activation_frequency_blacklist(
        d, n_sae, freq_min=None, freq_max=thr
    ).ids == oracle


def test_exact_boundary_excluded_then_included(tmp_path):
    # count == thr * n_segments exactly (50 == 0.5 * 100) -> excluded
    write_stats(
        tmp_path, n_segments=100, counts={3: 50, 9: 99}, n_sae_features=16
    )
    assert 3 not in fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5).ids
    write_stats(
        tmp_path, n_segments=100, counts={3: 51, 9: 99}, n_sae_features=16
    )
    assert 3 in fb.compute_activation_frequency_blacklist(tmp_path, 16, freq_min=None, freq_max=0.5).ids


def test_inactive_feature_only_killed_by_lower_bound(tmp_path):
    # feature 7 absent from stats (never fired). Under UB-only it is never
    # blacklisted (count=0 ≤ max*n is false unless max=0). Under LB≥0 the
    # dead feature IS killed. Pins the LB-vs-UB asymmetry on dead features.
    write_stats(tmp_path, n_segments=100, counts={0: 100}, n_sae_features=16)
    # UB-only: 7 survives at every threshold
    for thr in (0.001, 0.01, 0.5, 0.999):
        assert 7 not in fb.compute_activation_frequency_blacklist(
            tmp_path, 16, freq_min=None, freq_max=thr
        ).ids
    # LB at 0: 7 (dead) is killed; LB skipped: 7 stays
    assert 7 in fb.compute_activation_frequency_blacklist(
        tmp_path, 16, freq_min=0.0, freq_max=None
    ).ids
    assert 7 not in fb.compute_activation_frequency_blacklist(
        tmp_path, 16, freq_min=None, freq_max=None
    ).ids


@settings(max_examples=60, deadline=None)
@given(
    a=st.sampled_from([0.01, 1e-2, 0.010]),
    b=st.sampled_from([0.01, 1e-2, 0.010]),
)
def test_canonical_bound_path_invariance(tmp_path_factory, a, b):
    # bit-equal thresholds (0.01 == 1e-2 == 0.010) map to the same artifact
    d = tmp_path_factory.mktemp("c")
    assert fb.blacklist_paths(d, freq_min=None, freq_max=a) == fb.blacklist_paths(
        d, freq_min=None, freq_max=b
    )


def test_threshold_monotone_subset(tmp_path):
    """Higher threshold ⇒ blacklist is a subset of the lower-threshold one."""
    rng = random.Random(0)
    counts = {i: rng.randint(1, 1000) for i in range(60)}
    write_stats(tmp_path, n_segments=1000, counts=counts, n_sae_features=64)
    lo = set(fb.compute_activation_frequency_blacklist(tmp_path, 64, freq_min=None, freq_max=0.1).ids)
    hi = set(fb.compute_activation_frequency_blacklist(tmp_path, 64, freq_min=None, freq_max=0.5).ids)
    assert hi.issubset(lo)


@st.composite
def _bounded_corpus(draw):
    """Like _corpus, but draws an independent (freq_min, freq_max) pair (each
    optionally None) with freq_min < freq_max when both are set."""
    n_sae, n_segments, counts, _ = draw(_corpus())
    lo = draw(st.one_of(st.none(), st.floats(0.0, 0.98, allow_nan=False)))
    hi = draw(st.one_of(st.none(), st.floats(0.001, 1.0, allow_nan=False)))
    if lo is not None and hi is not None and lo >= hi:
        hi = None  # keep the kept-interval non-empty (lib would reject lo>=hi)
    return n_sae, n_segments, counts, lo, hi


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=_bounded_corpus())
def test_both_bounds_match_independent_oracle(tmp_path_factory, data):
    """Full two-bound predicate vs an independent oracle over the ENTIRE id
    space (so dead features killed by the LB are covered too)."""
    n_sae, n_segments, counts, lo, hi = data
    d = tmp_path_factory.mktemp("c")
    write_stats(d, n_segments=n_segments, counts=counts, n_sae_features=n_sae)
    if n_segments <= 0:
        oracle: tuple[int, ...] = ()
    else:
        lo_cut = None if lo is None else float(lo) * float(n_segments)
        hi_cut = None if hi is None else float(hi) * float(n_segments)
        oracle_list = []
        for fid in range(n_sae):
            c = float(counts.get(fid, 0))  # dead features: count 0
            kill_lo = lo_cut is not None and c <= lo_cut
            kill_hi = hi_cut is not None and c > hi_cut
            if kill_lo or kill_hi:
                oracle_list.append(fid)
        oracle = tuple(oracle_list)
    res = fb.compute_activation_frequency_blacklist(
        d, n_sae, freq_min=lo, freq_max=hi
    )
    assert res.ids == oracle


@settings(max_examples=80, deadline=None)
@given(x=st.one_of(st.none(), st.floats(0.0, 1.0, allow_nan=False)))
def test_bound_str_roundtrip(x):
    """``_bound_from_str(_bound_to_str(x)) == x`` for None and any [0,1]."""
    assert fb._bound_from_str(fb._bound_to_str(x)) == x
