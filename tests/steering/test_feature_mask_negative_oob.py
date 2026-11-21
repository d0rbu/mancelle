"""B5: deterministic membership for negative/OOB ids in keep_mask + loud assert.

Adversarial pre-merge audit found that keep_mask indexes its membership table
with raw feature_ids — a negative entry Python-wraps to member[hi-1] which is
always True (since ids.max() ∈ self.ids), yielding:
  - blacklist: keep=False for -1 (nondeterministic, wrong direction)
  - whitelist: keep=True for -1 (silent false-pass, I4 violation)
  - assert_no_masked with whitelist {5} and input [-1]: no raise (silent false-pass)

These tests are RED pre-fix (they document the exact breakage) and GREEN post-fix.

Structure:
  1. Wrap-determinism: keep_mask over [-1, 5, 7, 1000] with blacklist {3,7}
  2. assert_no_masked loud on negative: whitelist+neg → should raise (pre-fix: silent)
  3. Valid-id bit-identical regression pin: full arange(n) unchanged
  4. OOB id (>= max): not-member deterministically, no IndexError/wrap
  5. Goldens still hold (frozen hash values)
"""

from __future__ import annotations

import pytest
import torch as t

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask

# ---------------------------------------------------------------------------
# 1. Wrap-determinism
# ---------------------------------------------------------------------------

class TestKeepMaskNegativeOOB:
    """keep_mask must treat negative and OOB ids as not-a-member (deterministic)."""

    def test_blacklist_negative_id_is_not_member_deterministic(self):
        """Blacklist {3,7}: feature_id=-1 is NOT a member => keep=True (survives).

        Pre-fix: member[-1] = member[hi-1] = member[7] = True (wrap!) => keep=False
                 (wrong — a real id -1 is garbage, not a blacklisted id)
        Post-fix: -1 out-of-domain => is_member=False => keep=True (deterministic)
        """
        mask = FeatureMask("blacklist", (3, 7))
        fids = t.tensor([-1, 5, 7, 1000], dtype=t.int64)
        keep = mask.keep_mask(fids)
        # -1: not a member (out-of-domain negative) => not blacklisted => keep=True
        assert keep[0].item() is True, (
            f"keep_mask([-1]) with blacklist{{3,7}}: expected True (not-member), "
            f"got {keep[0].item()} — negative index wrapping detected (pre-fix)"
        )
        # 5: not a member of {3,7} => keep=True
        assert keep[1].item() is True, "keep_mask([5]) blacklist{3,7}: expected True"
        # 7: IS a member of {3,7} => blacklisted => keep=False
        assert keep[2].item() is False, "keep_mask([7]) blacklist{3,7}: expected False"
        # 1000: out-of-domain OOB => not a member => keep=True
        assert keep[3].item() is True, (
            f"keep_mask([1000]) blacklist{{3,7}}: expected True (OOB => not-member), "
            f"got {keep[3].item()}"
        )

    def test_whitelist_negative_id_is_not_member_deterministic(self):
        """Whitelist {5}: feature_id=-1 is NOT in whitelist => keep=False.

        Pre-fix: hi = max(-1 clipped? no — it uses max directly). Actually with
                 whitelist {5} and fids=[-1,5], hi = max(5,5)+1 = 6.
                 member[-1] = member[5] = True (wrap!) => keep=True for -1
                 (silent false-pass: invalid id appears whitelisted)
        Post-fix: -1 out-of-domain => is_member=False => keep=False (correct)
        """
        mask = FeatureMask("whitelist", (5,))
        fids = t.tensor([-1, 5], dtype=t.int64)
        keep = mask.keep_mask(fids)
        # -1: not in whitelist {5} => keep=False
        assert keep[0].item() is False, (
            f"keep_mask([-1]) with whitelist{{5}}: expected False (not-member), "
            f"got {keep[0].item()} — negative wrap false-pass detected (pre-fix)"
        )
        # 5: IS in whitelist => keep=True
        assert keep[1].item() is True, "keep_mask([5]) whitelist{5}: expected True"

    def test_oob_id_exceeds_max_is_not_member(self):
        """IDs > ids.max() are out-of-domain => not-a-member, no IndexError."""
        mask_bl = FeatureMask("blacklist", (3, 7))
        # OOB ids: 8, 100, 1000 — all > 7 (ids.max())
        fids = t.tensor([8, 100, 1000], dtype=t.int64)
        keep = mask_bl.keep_mask(fids)
        # All OOB => not blacklisted => keep=True
        assert keep.all().item() is True, (
            f"blacklist keep_mask for OOB ids [8,100,1000]: expected all True, got {keep.tolist()}"
        )

        mask_wl = FeatureMask("whitelist", (3, 7))
        keep_wl = mask_wl.keep_mask(fids)
        # All OOB => not whitelisted => keep=False
        assert not keep_wl.any().item(), (
            f"whitelist keep_mask for OOB ids [8,100,1000]: expected all False, got {keep_wl.tolist()}"
        )

    def test_mixed_negative_and_oob_no_index_error(self):
        """keep_mask must not raise IndexError for negative or large ids."""
        mask = FeatureMask("blacklist", (5,))
        fids = t.tensor([-100, -1, 0, 5, 6, 999], dtype=t.int64)
        # Should not raise; negative/OOB => not-member
        keep = mask.keep_mask(fids)
        assert keep.shape == (6,), f"unexpected shape {keep.shape}"
        # -100, -1: negative => not-member => blacklist keep=True
        assert keep[0].item() is True, "keep_mask([-100]) expected True"
        assert keep[1].item() is True, "keep_mask([-1]) expected True"
        # 0: valid, not in {5} => not-member => keep=True
        assert keep[2].item() is True, "keep_mask([0]) expected True"
        # 5: member of {5} => blacklisted => keep=False
        assert keep[3].item() is False, "keep_mask([5]) expected False"
        # 6: OOB (> ids.max()=5) => not-member => keep=True
        assert keep[4].item() is True, "keep_mask([6]) expected True"
        # 999: OOB => not-member => keep=True
        assert keep[5].item() is True, "keep_mask([999]) expected True"


# ---------------------------------------------------------------------------
# 2. assert_no_masked loud on negative ids
# ---------------------------------------------------------------------------

class TestAssertNoMaskedLoudOnNegative:
    """assert_no_masked must raise FeatureMaskError on negative feature ids.

    A negative feature id is definitionally invalid (real SAE ids are >= 0).
    Pre-fix: whitelist {5} + input [-1] => keep=True (wrap) => NO raise (silent).
    Post-fix: negative id => invalid => raise FeatureMaskError (loud).
    """

    def test_whitelist_negative_id_raises(self):
        """Whitelist {5} + assert_no_masked([-1]): must raise (negative is invalid).

        Pre-fix: -1 wraps to member[5]=True => keep=True => no raise (silent false-pass)
        Post-fix: raises FeatureMaskError (negative id is invalid)
        """
        mask = FeatureMask("whitelist", (5,))
        with pytest.raises(FeatureMaskError, match=r"(?i)negative|invalid"):
            mask.assert_no_masked(t.tensor([-1], dtype=t.int64))

    def test_blacklist_negative_id_raises_for_right_reason(self):
        """Blacklist {0} + assert_no_masked([-1]): must raise for the RIGHT reason.

        Pre-fix: member[-1] = member[ids.max()=0] = True => keep=~True=False =>
                 assert_no_masked raises (but for WRONG reason — wrap confusion,
                 not because -1 is invalid; it would NOT raise with blacklist {5}).
        Post-fix: raises explicitly because -1 is a negative (invalid) id,
                  regardless of mask contents.
        """
        mask = FeatureMask("blacklist", (0,))
        with pytest.raises(FeatureMaskError, match=r"(?i)negative|invalid"):
            mask.assert_no_masked(t.tensor([-1], dtype=t.int64))

    def test_blacklist_no_false_raise_for_right_reason(self):
        """Blacklist {5} + assert_no_masked([-1]): pre-fix would NOT raise (different wrap target).

        With blacklist {5}: member[-1] = member[hi-1], hi = max(-1, 5)+1 = 6
        But wait — max(feature_ids.max(), ids.max()) — feature_ids.max() = -1 (signed)
        in pre-fix code this could yield hi = max(-1,5)+1 = 6, member[5]=True,
        member[-1] = member[5] = True, keep=~True=False (blacklist drops -1).
        Pre-fix raises (wrong reason: treated as blacklisted member via wrap).
        Post-fix raises (right reason: negative id is invalid).
        In both cases it raises, but the post-fix message is deterministic/clear.
        """
        mask = FeatureMask("blacklist", (5,))
        with pytest.raises(FeatureMaskError, match=r"(?i)negative|invalid"):
            mask.assert_no_masked(t.tensor([-1], dtype=t.int64))

    def test_negative_with_valid_mix_raises(self):
        """Mixed tensor [-1, 3, 5]: the negative alone triggers the guard."""
        mask = FeatureMask("whitelist", (3, 5))
        with pytest.raises(FeatureMaskError):
            mask.assert_no_masked(t.tensor([-1, 3, 5], dtype=t.int64))

    def test_assert_no_masked_clean_valid_ids_no_raise(self):
        """Sanity: valid clean ids still don't raise (regression guard)."""
        mask = FeatureMask("blacklist", (9,))
        mask.assert_no_masked(t.tensor([1, 2, 3], dtype=t.int64))  # ok: no blacklisted id

        mask_wl = FeatureMask("whitelist", (1, 2, 3))
        mask_wl.assert_no_masked(t.tensor([1, 2, 3], dtype=t.int64))  # ok: all whitelisted

    def test_assert_no_masked_empty_no_raise(self):
        """Empty tensor: assert_no_masked never raises (existing contract)."""
        mask = FeatureMask("whitelist", (5,))
        mask.assert_no_masked(t.tensor([], dtype=t.int64))  # ok


# ---------------------------------------------------------------------------
# 3. Valid-id bit-identical regression pin
# ---------------------------------------------------------------------------

class TestValidIdBitIdentical:
    """Valid ids in [0, ids.max()] must produce bit-identical results post-fix.

    Ensures zero behavior change for the real production path.
    """

    def _expected_keep_blacklist(self, mask: FeatureMask, fids: t.Tensor) -> t.Tensor:
        """Reference implementation: keep = id not in mask.ids."""
        blocked = set(mask.ids)
        return t.tensor([int(v.item()) not in blocked for v in fids], dtype=t.bool)

    def _expected_keep_whitelist(self, mask: FeatureMask, fids: t.Tensor) -> t.Tensor:
        """Reference implementation: keep = id in mask.ids."""
        allowed = set(mask.ids)
        return t.tensor([int(v.item()) in allowed for v in fids], dtype=t.bool)

    def test_blacklist_valid_range_bit_identical(self):
        """Blacklist {1, 4, 7}: keep_mask over arange(10) bit-identical to reference."""
        mask = FeatureMask("blacklist", (1, 4, 7))
        fids = t.arange(0, 10, dtype=t.int64)
        got = mask.keep_mask(fids)
        expected = self._expected_keep_blacklist(mask, fids)
        assert t.equal(got, expected), (
            f"blacklist keep_mask mismatch:\n  got      {got.tolist()}\n"
            f"  expected {expected.tolist()}"
        )

    def test_whitelist_valid_range_bit_identical(self):
        """Whitelist {2, 5, 8}: keep_mask over arange(10) bit-identical to reference."""
        mask = FeatureMask("whitelist", (2, 5, 8))
        fids = t.arange(0, 10, dtype=t.int64)
        got = mask.keep_mask(fids)
        expected = self._expected_keep_whitelist(mask, fids)
        assert t.equal(got, expected), (
            f"whitelist keep_mask mismatch:\n  got      {got.tolist()}\n"
            f"  expected {expected.tolist()}"
        )

    def test_allowed_universe_mask_bit_identical(self):
        """allowed_universe_mask over full n_sae=20 is unchanged."""
        n = 20
        bl = FeatureMask("blacklist", (1, 3, 7))
        got_bl = bl.allowed_universe_mask(n)
        expected_bl = t.tensor(
            [i not in {1, 3, 7} for i in range(n)], dtype=t.bool
        )
        assert t.equal(got_bl, expected_bl), (
            "allowed_universe_mask blacklist mismatch"
        )

        wl = FeatureMask("whitelist", (1, 3, 7))
        got_wl = wl.allowed_universe_mask(n)
        expected_wl = t.tensor(
            [i in {1, 3, 7} for i in range(n)], dtype=t.bool
        )
        assert t.equal(got_wl, expected_wl), (
            "allowed_universe_mask whitelist mismatch"
        )

    def test_apply_to_positive_valid_ids_bit_identical(self):
        """apply_to_positive with valid ids unchanged: blacklist drops correctly."""
        mask = FeatureMask("blacklist", (2, 4))
        fids = t.tensor([1, 2, 3, 4, 5], dtype=t.int64)
        mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out_fids, out_mags = mask.apply_to_positive(fids, mags)
        assert out_fids.tolist() == [1, 3, 5], f"unexpected {out_fids.tolist()}"
        assert out_mags.tolist() == [1.0, 3.0, 5.0], f"unexpected {out_mags.tolist()}"

    def test_assert_no_masked_valid_ids_unchanged(self):
        """assert_no_masked: no raise for clean valid ids (both modes) — regression."""
        # Blacklist: clean ids don't raise
        FeatureMask("blacklist", (5,)).assert_no_masked(t.tensor([1, 2, 3, 4], dtype=t.int64))
        # Blacklist: blacklisted id raises (existing behavior)
        with pytest.raises(FeatureMaskError):
            FeatureMask("blacklist", (3,)).assert_no_masked(t.tensor([1, 2, 3, 4], dtype=t.int64))
        # Whitelist: all-in-whitelist doesn't raise
        FeatureMask("whitelist", (1, 2)).assert_no_masked(t.tensor([1, 2], dtype=t.int64))
        # Whitelist: non-whitelisted raises
        with pytest.raises(FeatureMaskError):
            FeatureMask("whitelist", (1,)).assert_no_masked(t.tensor([1, 2], dtype=t.int64))


# ---------------------------------------------------------------------------
# 4. OOB id (>= n_sae / >= max) — deterministic not-member, no crash
# ---------------------------------------------------------------------------

class TestOOBIdDeterministic:
    """OOB ids (>= ids.max()+1) are not-a-member deterministically."""

    def test_whitelist_oob_keep_false(self):
        """Whitelist {3}: ids 4, 5, 100 are OOB => not whitelisted => keep=False."""
        mask = FeatureMask("whitelist", (3,))
        n = 50  # use as universe size hint
        fids = t.tensor([n, n + 50], dtype=t.int64)
        keep = mask.keep_mask(fids)
        assert keep[0].item() is False, "whitelist OOB keep_mask[n]: expected False"
        assert keep[1].item() is False, "whitelist OOB keep_mask[n+50]: expected False"

    def test_blacklist_oob_keep_true(self):
        """Blacklist {3}: ids > 3 are OOB => not blacklisted => keep=True."""
        mask = FeatureMask("blacklist", (3,))
        n = 50
        fids = t.tensor([n, n + 50], dtype=t.int64)
        keep = mask.keep_mask(fids)
        assert keep[0].item() is True, "blacklist OOB keep_mask[n]: expected True"
        assert keep[1].item() is True, "blacklist OOB keep_mask[n+50]: expected True"

    def test_oob_assert_no_masked_no_raise_for_blacklist(self):
        """OOB ids in a blacklist are not-a-member => assert_no_masked passes (no raise)."""
        mask = FeatureMask("blacklist", (3,))
        # 1000 is OOB: not blacklisted => assert passes
        mask.assert_no_masked(t.tensor([1000], dtype=t.int64))

    def test_oob_assert_no_masked_raises_for_whitelist(self):
        """OOB ids in a whitelist are not-whitelisted => assert_no_masked raises."""
        mask = FeatureMask("whitelist", (3,))
        # 1000 is OOB: not whitelisted => assert raises (I4 violation)
        with pytest.raises(FeatureMaskError):
            mask.assert_no_masked(t.tensor([1000], dtype=t.int64))


# ---------------------------------------------------------------------------
# 5. Frozen hash goldens still hold
# ---------------------------------------------------------------------------

class TestFrozenHashGoldens:
    """The frozen canonical_dict / compute_selection_hash goldens are unchanged.

    These are the I1 invariants. canonical_dict must not be touched.
    Goldens: ae63e39f... and f8bfb3a1...
    These are tested more thoroughly in test_canonical_feature_selection_hash.py,
    test_feature_mask_integration.py, test_selection_hash_feature_mask.py.
    Here we just confirm canonical_dict still returns plain-str mode and sorted ids.
    """

    def test_canonical_dict_mode_is_plain_str(self):
        """canonical_dict returns mode as plain str (not FeatureMaskMode enum)."""
        m = FeatureMask("blacklist", (1, 3))
        d = m.canonical_dict()
        assert type(d["mode"]) is str, f"mode type {type(d['mode'])} is not str"
        assert d["mode"] == "blacklist"

    def test_canonical_dict_ids_sorted(self):
        """canonical_dict ids are sorted regardless of input order."""
        m = FeatureMask("blacklist", (5, 1, 3))
        d = m.canonical_dict()
        assert d["ids"] == [1, 3, 5], f"canonical ids not sorted: {d['ids']}"

    def test_canonical_dict_unchanged_by_fix(self):
        """canonical_dict output for two golden masks matches expected values."""
        m1 = FeatureMask("blacklist", (100, 200, 300))
        d1 = m1.canonical_dict()
        assert d1 == {"mode": "blacklist", "ids": [100, 200, 300]}

        m2 = FeatureMask("whitelist", (42,))
        d2 = m2.canonical_dict()
        assert d2 == {"mode": "whitelist", "ids": [42]}
