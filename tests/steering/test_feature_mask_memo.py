# tests/steering/test_feature_mask_memo.py
"""Byte-identity + safety tests for the FeatureMask derived-tensor memo (#740).

The memo caches the constant tensors (`t.tensor(self.ids)` + full-width
membership) that were previously rebuilt on every one of the 45k+ per-segment
`feature_mask` calls in the splice preselection hot loop. Every cached value
must be BYTE-IDENTICAL to the pre-memo per-call computation, the cache must not
be corruptible by callers mutating returned tensors, and mask identity
(==/hash/pickle) must be unaffected by cache state.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import threading

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode

# ---------------------------------------------------------------------------
# Reference oracle: a faithful copy of the PRE-memo per-call implementation.
# The memoized methods must equal these exactly.
# ---------------------------------------------------------------------------


def ref_membership(ids: tuple[int, ...], n: int) -> t.Tensor:
    mem = t.zeros(n, dtype=t.bool)
    idx = t.tensor(ids, dtype=t.int64)
    if int(idx.max().item()) >= n:
        raise FeatureMaskError(f"feature_mask id {int(idx.max().item())} >= n_sae_features {n}")
    mem[idx] = True
    return mem


def ref_allowed(mode: str, ids: tuple[int, ...], n: int) -> t.Tensor:
    member = ref_membership(ids, n)
    return ~member if mode == "blacklist" else member


def ref_keep(mode: str, ids: tuple[int, ...], fids: t.Tensor) -> t.Tensor:
    if fids.numel() == 0:
        return t.zeros(0, dtype=t.bool, device=fids.device)
    idst = t.tensor(ids, dtype=t.int64, device=fids.device)
    max_id = int(idst.max().item())
    member = t.zeros(max_id + 1, dtype=t.bool, device=fids.device)
    member[idst] = True
    fl = fids.long()
    safe = (fl >= 0) & (fl <= max_id)
    is_member = t.zeros(fl.shape, dtype=t.bool, device=fids.device)
    is_member[safe] = member[fl[safe]]
    return ~is_member if mode == "blacklist" else is_member


N = 65536
MASKS = [
    ("whitelist", (5, 10, 20)),
    ("blacklist", (0, 1, 2, 63)),
    ("whitelist", (0,)),
    ("blacklist", (65535,)),
    ("whitelist", (7, 99, 12345, 65535)),
    ("whitelist", tuple(range(0, 4000, 7))),
]
FID_SETS = [
    t.tensor([], dtype=t.int64),
    t.tensor([5, 6, 10, 21, 20, 0], dtype=t.int64),
    t.tensor([-1, 5, 999999, 10, -7], dtype=t.int64),  # negative + OOB kept safe
    t.tensor([65535, 65534, 0, 1], dtype=t.int64),
    t.tensor([20, 20, 20], dtype=t.int64),  # duplicates
]


@pytest.mark.parametrize("mode,ids", MASKS)
def test_allowed_universe_mask_byte_identical(mode, ids):
    m = FeatureMask(mode, ids)
    exp = ref_allowed(mode, ids, N)
    for _ in range(3):
        assert t.equal(m.allowed_universe_mask(N), exp)


@pytest.mark.parametrize("mode,ids", MASKS)
@pytest.mark.parametrize("fids", FID_SETS)
def test_keep_mask_byte_identical(mode, ids, fids):
    m = FeatureMask(mode, ids)
    exp = ref_keep(mode, ids, fids)
    for _ in range(3):
        assert t.equal(m.keep_mask(fids), exp)


@pytest.mark.parametrize("mode,ids", MASKS)
@pytest.mark.parametrize("fids", FID_SETS)
def test_apply_to_positive_byte_identical(mode, ids, fids):
    m = FeatureMask(mode, ids)
    keep = ref_keep(mode, ids, fids)
    mags = fids.float() * 2.0
    exp_ids, exp_mags = fids[keep], mags[keep]
    got_ids, got_mags = m.apply_to_positive(fids, mags)
    assert t.equal(got_ids, exp_ids) and t.equal(got_mags, exp_mags)


def test_returned_tensors_are_fresh_not_aliased():
    """Two calls must return DISTINCT storage (no shared cached object leaks)."""
    m = FeatureMask("whitelist", (1, 2, 3))
    a2, b2 = m.allowed_universe_mask(N), m.allowed_universe_mask(N)
    assert a2.data_ptr() != b2.data_ptr()
    k1 = m.keep_mask(t.tensor([1, 4], dtype=t.int64))
    k2 = m.keep_mask(t.tensor([1, 4], dtype=t.int64))
    assert k1.data_ptr() != k2.data_ptr()


@pytest.mark.parametrize("mode,ids", MASKS)
def test_mutating_returned_tensor_cannot_corrupt_cache(mode, ids):
    m = FeatureMask(mode, ids)
    exp_allowed = ref_allowed(mode, ids, N)
    # Aggressively mutate every returned tensor, then re-read.
    m.allowed_universe_mask(N).fill_(True)
    m.keep_mask(t.tensor([ids[0]], dtype=t.int64)).zero_()
    assert t.equal(m.allowed_universe_mask(N), exp_allowed)


def test_allowed_universe_oob_raises_with_exact_message():
    m = FeatureMask("whitelist", (5, 100))
    with pytest.raises(FeatureMaskError, match=r"feature_mask id 100 >= n_sae_features 50"):
        m.allowed_universe_mask(50)


def test_multiple_n_values_each_correct_and_independent():
    m = FeatureMask("whitelist", (3, 300))
    assert t.equal(
        m.allowed_universe_mask(1000),
        ref_allowed("whitelist", (3, 300), 1000),
    )
    assert t.equal(
        m.allowed_universe_mask(500),
        ref_allowed("whitelist", (3, 300), 500),
    )
    # A too-small n still raises even after larger n cached.
    with pytest.raises(FeatureMaskError):
        m.allowed_universe_mask(100)
    # ...and the previously-cached n is still correct.
    assert t.equal(
        m.allowed_universe_mask(1000),
        ref_allowed("whitelist", (3, 300), 1000),
    )


def test_keep_mask_empty_input_matches_and_no_cache_needed():
    m = FeatureMask("blacklist", (1, 2))
    got = m.keep_mask(t.tensor([], dtype=t.int64))
    assert got.numel() == 0 and got.dtype == t.bool


def test_assert_no_masked_still_guards():
    wl = FeatureMask("whitelist", (1, 2, 3))
    wl.assert_no_masked(t.tensor([1, 3], dtype=t.int64))  # ok
    with pytest.raises(FeatureMaskError, match="feature_mask violated"):
        wl.assert_no_masked(t.tensor([1, 9], dtype=t.int64))
    with pytest.raises(FeatureMaskError, match="negative feature id"):
        wl.assert_no_masked(t.tensor([1, -5], dtype=t.int64))


# ---------------------------------------------------------------------------
# Identity / pickle / copy: the memo must be invisible to mask identity.
# ---------------------------------------------------------------------------


def test_equality_and_hash_unaffected_by_cache_state():
    a = FeatureMask("whitelist", (2, 1, 3))
    b = FeatureMask("whitelist", (3, 2, 1))  # same after canonicalization
    a.allowed_universe_mask(N)  # warm a's cache only
    a.keep_mask(t.tensor([1], dtype=t.int64))
    assert a == b
    assert hash(a) == hash(b)


def test_memo_excluded_from_repr():
    m = FeatureMask("whitelist", (1, 2))
    assert "_memo" not in repr(m)
    assert "_MaskMemo" not in repr(m)


def test_pickle_roundtrip_preserves_identity_and_correctness():
    m = FeatureMask("whitelist", (4, 8, 15, 16, 23, 42))
    m.allowed_universe_mask(N)  # warm the cache before pickling
    r = pickle.loads(pickle.dumps(m))
    assert r == m and hash(r) == hash(m)
    assert t.equal(
        r.allowed_universe_mask(N),
        ref_allowed("whitelist", m.ids, N),
    )
    assert t.equal(r.keep_mask(FID_SETS[1]), ref_keep("whitelist", m.ids, FID_SETS[1]))


def test_pickle_does_not_serialize_cached_tensors():
    """The wire payload for a warm mask must be no larger than a cold one
    (cache dropped on pickle) — guards against shipping GBs of tensors when a
    mask is pickled for multiprocessing."""
    cold = FeatureMask("whitelist", tuple(range(0, 4000, 3)))
    warm = FeatureMask("whitelist", tuple(range(0, 4000, 3)))
    warm.allowed_universe_mask(N)
    warm._allowed_feature_ids_cached(N)
    warm.keep_mask(t.arange(100))
    assert len(pickle.dumps(warm)) == len(pickle.dumps(cold))


@pytest.mark.parametrize("mode,ids", MASKS)
def test_internal_allowed_views_are_cached_but_public_results_are_fresh(mode, ids):
    mask = FeatureMask(mode, ids)
    internal_a = mask._allowed_universe_cached(N)
    internal_b = mask._allowed_universe_cached(N)
    ids_a = mask._allowed_feature_ids_cached(N)
    ids_b = mask._allowed_feature_ids_cached(N)
    assert internal_a.data_ptr() == internal_b.data_ptr()
    assert ids_a.data_ptr() == ids_b.data_ptr()
    assert t.equal(internal_a, ref_allowed(mode, ids, N))
    assert t.equal(ids_a, t.nonzero(internal_a, as_tuple=False).squeeze(1))

    public = mask.allowed_universe_mask(N)
    assert public.data_ptr() != internal_a.data_ptr()
    public.logical_not_()
    assert t.equal(mask._allowed_universe_cached(N), ref_allowed(mode, ids, N))


def test_deepcopy_and_replace_get_fresh_correct_memo():
    m = FeatureMask("blacklist", (1, 2, 3))
    m.allowed_universe_mask(N)
    d = copy.deepcopy(m)
    assert d == m
    assert t.equal(d.allowed_universe_mask(N), ref_allowed("blacklist", m.ids, N))
    r = dataclasses.replace(m)
    assert r == m
    assert t.equal(r.keep_mask(FID_SETS[1]), ref_keep("blacklist", m.ids, FID_SETS[1]))


def test_still_frozen():
    m = FeatureMask("whitelist", (1, 2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.mode = FeatureMaskMode.blacklist  # type: ignore[misc]  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.ids = (9,)  # type: ignore[misc]  # ty: ignore[invalid-assignment]


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode,ids", MASKS[:4])
def test_keep_mask_cuda_matches_cpu(mode, ids):
    m = FeatureMask(mode, ids)
    fids = t.tensor([ids[0], 5, 999999, -1], dtype=t.int64)
    cpu = m.keep_mask(fids)
    gpu = m.keep_mask(fids.cuda())
    assert t.equal(gpu.cpu(), cpu)


# ---------------------------------------------------------------------------
# Regression tests for the adversarial-review findings (fail-loud + parity in
# the error/edge paths, not just the happy path).
# ---------------------------------------------------------------------------


def test_oob_raises_before_allocating_huge_buffer():
    """Finding 1: a corrupt/huge id with a small n must raise FeatureMaskError
    via the precomputed max id BEFORE trying to allocate a max_id+1 buffer
    (which would OOM). This test would hang/OOM if the guard came after alloc."""
    m = FeatureMask("whitelist", (2**40,))  # 1 TiB bool buffer if allocated
    with pytest.raises(FeatureMaskError, match=r">= n_sae_features 1"):
        m.allowed_universe_mask(1)


def test_warm_cache_rejects_non_int_n():
    """Finding 4: after caching allowed_universe_mask(N), a bool/float n that is
    dict-equal to a cached int key must NOT alias to the cached tensor — it
    fails loud (pre-memo t.zeros rejected non-int n)."""
    m = FeatureMask("whitelist", (1, 2, 3))
    m.allowed_universe_mask(N)  # warm the int key
    for bad in (True, 1.0, N * 1.0):
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            m.allowed_universe_mask(bad)  # type: ignore[arg-type]


def test_asdict_excludes_memo_and_is_json_serializable():
    """Finding 5: the memo/_max_id are NOT dataclass fields, so dataclasses
    introspection is behavior-compatible with the pre-change mask."""
    m = FeatureMask("whitelist", (5, 1, 9))
    m.allowed_universe_mask(N)  # warm — cached tensors must not leak into asdict
    m.keep_mask(t.tensor([1], dtype=t.int64))
    d = dataclasses.asdict(m)
    assert set(d) == {"mode", "ids"}
    assert [f.name for f in dataclasses.fields(m)] == ["mode", "ids"]
    json.dumps({"mode": str(d["mode"]), "ids": list(d["ids"])})  # must not raise


def test_cold_cache_is_thread_safe():
    """Finding 3: concurrent first calls on a COLD mask must not observe a
    half-initialised cache (no AssertionError / no wrong value). _max_id is
    precomputed and `member` is published last, so this is race-free."""
    exc: list[BaseException] = []
    for _ in range(20):  # repeat to shake out races
        m = FeatureMask("whitelist", tuple(range(0, 3000, 7)))
        exp_keep = ref_keep("whitelist", m.ids, FID_SETS[1])
        barrier = threading.Barrier(8)

        def worker(
            *,
            barrier=barrier,
            mask=m,
            expected_keep=exp_keep,
        ) -> None:
            try:
                barrier.wait()
                assert t.equal(mask.keep_mask(FID_SETS[1]), expected_keep)
            except BaseException as e:  # noqa: BLE001
                exc.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert not exc, f"thread-safety violation: {exc[:3]}"


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_allowed_universe_tracks_ambient_default_device():
    """Finding 2: under a non-CPU ambient torch default device the result must
    be built on that device (not a stale CPU-cached tensor)."""
    m = FeatureMask("whitelist", (1, 2, 3))
    cpu = m.allowed_universe_mask(N)  # warm the CPU cache first
    assert cpu.device.type == "cpu"
    with t.device("cuda"):
        got = m.allowed_universe_mask(N)
    assert got.device.type == "cuda"
    assert t.equal(got.cpu(), cpu)
