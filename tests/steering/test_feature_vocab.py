"""Unit tests for FeatureVocab — the encoder's compacted feature dictionary (#695 PR A).

FeatureVocab is the primitive the whole compaction rests on. Its single load-bearing
invariant is:

    FeatureVocab.from_mask(mask, n_sae) is a PURE, DETERMINISTIC function of
    (mask, n_sae).

That is what lets the existing ``bl_``/``wl_`` checkpoint leaf uniquely identify the
vocab without any new hash component (spec §4.1).

These tests pin the primitive in isolation: no model, no checkpoint, no torch.compile.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from errors import FeatureMaskError, FeatureVocabError, NanoGPTError
from steering.conditional import feature_vocab as feature_vocab_module
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab

N = 16  # tiny original dictionary width used throughout


def _vocab(ids=(0, 3, 7, 11), n=N) -> FeatureVocab:
    return FeatureVocab(n_sae_features=n, original_ids=tuple(ids))


# ===========================================================================
# 1. Construction + validation
# ===========================================================================


class TestConstruction:
    def test_valid_construction(self):
        v = _vocab((0, 3, 7, 11))
        assert v.n_sae_features == N
        assert v.original_ids == (0, 3, 7, 11)

    def test_canonicalizes_to_sorted_unique(self):
        """Mirrors FeatureMask.__post_init__: any source order/dup ⇒ same canonical form."""
        v = FeatureVocab(n_sae_features=N, original_ids=(7, 3, 7, 0, 11, 3))
        assert v.original_ids == (0, 3, 7, 11)

    def test_two_logically_identical_vocabs_are_equal_and_hash_alike(self):
        a = FeatureVocab(n_sae_features=N, original_ids=(7, 0, 3))
        b = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7))
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_is_frozen(self):
        v = _vocab()
        with pytest.raises((AttributeError, TypeError)):
            v.n_sae_features = 99  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    @pytest.mark.parametrize(
        "bad_ids",
        [(), (-1, 3), (0, 3.0), (0, "3"), (0, None), (0, True)],
    )
    def test_refined_id_domain_rejected(self, bad_ids):
        with pytest.raises(BeartypeCallHintParamViolation):
            FeatureVocab(n_sae_features=N, original_ids=bad_ids)

    def test_id_at_or_above_n_sae_rejected(self):
        with pytest.raises(FeatureVocabError, match="out of range"):
            FeatureVocab(n_sae_features=N, original_ids=(0, N))

    @pytest.mark.parametrize("bad_n", [0, -1, True, 3.0])
    def test_refined_width_domain_rejected(self, bad_n):
        with pytest.raises(BeartypeCallHintParamViolation):
            FeatureVocab(n_sae_features=bad_n, original_ids=(0,))

    def test_error_is_a_nanogpt_error_but_NOT_a_feature_mask_error(self):
        """Spec §9b: sibling of FeatureMaskError, not a subclass.

        An `except FeatureMaskError` handler must NOT silently swallow vocab errors —
        an out-of-vocabulary id and an invalid mask are different failures.
        """
        assert issubclass(FeatureVocabError, NanoGPTError)
        assert not issubclass(FeatureVocabError, FeatureMaskError)


# ===========================================================================
# 2. size / is_identity
# ===========================================================================


class TestSizeAndIdentity:
    def test_size_is_number_of_rows(self):
        assert _vocab((0, 3, 7, 11)).size == 4

    def test_is_identity_true_for_full_range(self):
        assert FeatureVocab(n_sae_features=4, original_ids=(0, 1, 2, 3)).is_identity

    def test_is_identity_false_when_any_row_missing(self):
        assert not FeatureVocab(n_sae_features=4, original_ids=(0, 1, 3)).is_identity


# ===========================================================================
# 3. from_mask — THE load-bearing invariant
# ===========================================================================


class TestFromMask:
    def test_none_mask_means_no_compaction(self):
        assert FeatureVocab.from_mask(None, N) is None

    def test_whitelist_keeps_only_listed_ids(self):
        m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 5, 9))
        v = FeatureVocab.from_mask(m, N)
        assert v is not None
        assert v.original_ids == (1, 5, 9)
        assert v.n_sae_features == N

    def test_blacklist_keeps_the_complement(self):
        m = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(1, 5, 9))
        v = FeatureVocab.from_mask(m, N)
        assert v is not None
        assert v.original_ids == tuple(i for i in range(N) if i not in (1, 5, 9))
        assert v.size == N - 3

    def test_vocab_equals_the_masks_allowed_universe_exactly(self):
        """The vocab IS allowed_universe_mask().nonzero() — for BOTH modes."""
        for mode in (FeatureMaskMode.blacklist, FeatureMaskMode.whitelist):
            m = FeatureMask(mode=mode, ids=(2, 4, 6))
            v = FeatureVocab.from_mask(m, N)
            assert v is not None
            allowed = m.allowed_universe_mask(N)
            expected = tuple(int(i) for i in allowed.nonzero().flatten().tolist())
            assert v.original_ids == expected

    def test_from_mask_is_a_pure_deterministic_function(self):
        """Same (mask, n_sae) ⇒ identical vocab. This is what lets the existing
        bl_/wl_ checkpoint leaf pin the vocab with NO new hash component (spec §4.1)."""
        m1 = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(9, 5, 1))
        m2 = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(1, 5, 9, 5))
        assert m1 == m2  # FeatureMask canonicalizes
        assert FeatureVocab.from_mask(m1, N) == FeatureVocab.from_mask(m2, N)

    def test_identity_result_is_normalized_to_none(self):
        """Spec §9b: an identity vocab normalizes to None so the byte-identical
        (zero-added-ops) path cannot be entered accidentally."""
        # A blacklist over an empty complement is impossible, but a whitelist of
        # every id is constructible and IS the identity.
        m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(N)))
        assert FeatureVocab.from_mask(m, N) is None

    def test_empty_allowed_universe_raises(self):
        """A blacklist of every id leaves nothing to embed."""
        m = FeatureMask(mode=FeatureMaskMode.blacklist, ids=tuple(range(N)))
        with pytest.raises(FeatureVocabError, match="zero features"):
            FeatureVocab.from_mask(m, N)

    def test_mask_id_beyond_n_sae_raises(self):
        m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, N + 5))
        with pytest.raises((FeatureVocabError, FeatureMaskError)):
            FeatureVocab.from_mask(m, N)

    def test_single_feature_vocab_is_legal(self):
        m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(7,))
        v = FeatureVocab.from_mask(m, N)
        assert v is not None and v.size == 1 and v.original_ids == (7,)


# ===========================================================================
# 4. Tensor maps
# ===========================================================================


class TestTensorMaps:
    def test_old_to_new_tensor_shape_dtype_and_content(self):
        v = _vocab((0, 3, 7, 11))
        o2n = v.old_to_new_tensor()
        assert o2n.shape == (N,)
        assert o2n.dtype == t.int64
        assert o2n[0].item() == 0
        assert o2n[3].item() == 1
        assert o2n[7].item() == 2
        assert o2n[11].item() == 3

    def test_old_to_new_uses_minus_one_for_out_of_vocab(self):
        v = _vocab((0, 3, 7, 11))
        o2n = v.old_to_new_tensor()
        for oov in (1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15):
            assert o2n[oov].item() == -1, f"id {oov} should be out-of-vocab"

    def test_new_to_old_tensor_shape_dtype_and_content(self):
        v = _vocab((0, 3, 7, 11))
        n2o = v.new_to_old_tensor()
        assert n2o.shape == (4,)
        assert n2o.dtype == t.int64
        assert n2o.tolist() == [0, 3, 7, 11]

    def test_the_two_tensors_are_exact_inverses(self):
        v = _vocab((0, 3, 7, 11))
        o2n, n2o = v.old_to_new_tensor(), v.new_to_old_tensor()
        # o2n[n2o[j]] == j  for every compact row j
        assert t.equal(o2n[n2o], t.arange(v.size, dtype=t.int64))

    def test_tensors_honour_the_device_kwarg(self):
        v = _vocab()
        assert v.old_to_new_tensor(device="cpu").device.type == "cpu"
        assert v.new_to_old_tensor(device="cpu").device.type == "cpu"


# ===========================================================================
# 5. to_compact / to_original
# ===========================================================================


class TestIdTranslation:
    def test_round_trip_holds_for_every_vocab_id(self):
        """Issue #695 requirement: new→old(old→new(id)) == id for every whitelisted id."""
        v = _vocab((0, 3, 7, 11))
        ids = t.tensor(v.original_ids, dtype=t.int64)
        assert t.equal(v.to_original(v.to_compact(ids)), ids)

    def test_to_compact_RAISES_on_an_in_range_out_of_vocab_id(self):
        """An id the mask excluded has NO embedding row, so there is nothing to return.

        This once returned the ``-1`` sentinel under an opt-in. The option is gone — see
        ``TestToCompactAlwaysRaisesOnOutOfVocab`` for why ``-1`` is lethal. The CASE is
        unchanged (1 and 2 are legal SAE ids that this vocab excludes); only the outcome
        is, and the error names EXACTLY the two offenders — not the in-vocab ids that
        happened to share the tensor with them.
        """
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([0, 1, 3, 2], dtype=t.int64))
        assert "[1, 2]" in str(exc.value), f"offenders misreported: {exc.value}"

    def test_to_compact_preserves_shape(self):
        v = _vocab((0, 3, 7, 11))
        out = v.to_compact(t.tensor([[0, 3], [7, 11]], dtype=t.int64))
        assert out.shape == (2, 2)
        assert out.tolist() == [[0, 1], [2, 3]]

    def test_to_compact_rejects_ids_beyond_n_sae(self):
        v = _vocab()
        with pytest.raises(FeatureVocabError, match="out of range"):
            v.to_compact(t.tensor([0, N], dtype=t.int64))

    def test_to_compact_rejects_negative_ids(self):
        v = _vocab()
        with pytest.raises(FeatureVocabError, match="out of range"):
            v.to_compact(t.tensor([0, -1], dtype=t.int64))

    def test_to_original_rejects_out_of_bounds_compact_ids(self):
        v = _vocab((0, 3, 7, 11))  # V == 4
        with pytest.raises(FeatureVocabError, match="out of range"):
            v.to_original(t.tensor([0, 4], dtype=t.int64))

    def test_empty_tensors_round_trip_without_error(self):
        v = _vocab()
        empty = t.tensor([], dtype=t.int64)
        assert v.to_compact(empty).numel() == 0
        assert v.to_original(empty).numel() == 0


class TestIdTranslationDtypes:
    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            (t.float32, 3.9),
            (t.float64, 7.9),
            (t.complex64, 11 + 2j),
            (t.complex128, 3 + 2j),
            (t.bool, False),
        ],
    )
    def test_to_compact_rejects_non_integer_dtype(self, dtype, value):
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc_info:
            v.to_compact(t.tensor([value], dtype=dtype))
        assert "integer tensor" in str(exc_info.value)
        assert str(dtype) in str(exc_info.value)

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            (t.float32, 2.9),
            (t.float64, 3.9),
            (t.complex64, 1 + 2j),
            (t.complex128, 2 + 2j),
            (t.bool, False),
        ],
    )
    def test_to_original_rejects_non_integer_dtype(self, dtype, value):
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc_info:
            v.to_original(t.tensor([value], dtype=dtype))
        assert "integer tensor" in str(exc_info.value)
        assert str(dtype) in str(exc_info.value)

    @pytest.mark.parametrize(
        ("method", "truncates_valid", "truncates_invalid"),
        [
            ("to_compact", 3.9, 5.9),
            ("to_original", 1.9, 4.9),
        ],
    )
    def test_non_integer_values_fail_for_identical_dtype_reason(
        self, method, truncates_valid, truncates_invalid
    ):
        translate = getattr(_vocab((0, 3, 7, 11)), method)
        messages = []
        for value in (truncates_valid, truncates_invalid):
            with pytest.raises(FeatureVocabError) as exc_info:
                translate(t.tensor([value], dtype=t.float32))
            messages.append(str(exc_info.value))
        assert messages[0] == messages[1]
        assert "torch.float32" in messages[0]

    @pytest.mark.parametrize("method", ["to_compact", "to_original"])
    @pytest.mark.parametrize("dtype", [t.float32, t.complex64, t.bool])
    def test_shaped_empty_non_integer_tensor_raises_before_fast_path(
        self, method, dtype
    ):
        translate = getattr(_vocab((0, 3, 7, 11)), method)
        ids = t.empty((2, 0, 3), dtype=dtype)
        with pytest.raises(FeatureVocabError) as exc_info:
            translate(ids)
        assert str(dtype) in str(exc_info.value)

    @pytest.mark.parametrize(
        "dtype",
        [
            t.uint8,
            t.int8,
            t.int16,
            t.int32,
            t.int64,
            t.uint16,
            t.uint32,
            t.uint64,
        ],
    )
    def test_every_integer_dtype_preserves_values_shape_device_and_int64_output(
        self, dtype
    ):
        v = _vocab((0, 3, 7, 11))
        original = t.tensor([[0, 3], [7, 11]], dtype=dtype)
        compact = v.to_compact(original)
        assert compact.shape == original.shape
        assert compact.device == original.device
        assert compact.dtype == t.int64
        assert t.equal(compact, t.tensor([[0, 1], [2, 3]], dtype=t.int64))

        round_trip = v.to_original(compact)
        assert round_trip.shape == original.shape
        assert round_trip.device == original.device
        assert round_trip.dtype == t.int64
        assert t.equal(round_trip, original.to(t.int64))

    @pytest.mark.parametrize(
        "dtype",
        [
            t.uint8,
            t.int8,
            t.int16,
            t.int32,
            t.int64,
            t.uint16,
            t.uint32,
            t.uint64,
        ],
    )
    def test_shaped_empty_integer_fast_path_preserves_shape_device_and_upcast(
        self, dtype
    ):
        v = _vocab((0, 3, 7, 11))
        empty = t.empty((2, 0, 3), dtype=dtype)
        for translated in (v.to_compact(empty), v.to_original(empty)):
            assert translated.shape == empty.shape
            assert translated.device == empty.device
            assert translated.dtype == t.int64


# ===========================================================================
# 6. select_rows — the slice-init + W_dec-subset primitive (spec §1.4, §4.8)
# ===========================================================================


class TestSelectRows:
    def test_select_rows_takes_exactly_the_vocab_rows(self):
        v = _vocab((0, 3, 7, 11))
        full = t.arange(N * 2, dtype=t.float32).reshape(N, 2)
        got = v.select_rows(full)
        assert got.shape == (4, 2)
        assert t.equal(got, full[t.tensor([0, 3, 7, 11])])

    def test_select_rows_is_bitwise_exact(self):
        """This is what makes slice-init byte-exact: compact[j] IS full[original_ids[j]],
        bit for bit — no arithmetic, just a gather."""
        v = _vocab((0, 3, 7, 11))
        full = t.randn(N, 5, dtype=t.float32)
        got = v.select_rows(full)
        for j, orig in enumerate(v.original_ids):
            assert t.equal(got[j], full[orig])

    def test_select_rows_preserves_dtype(self):
        v = _vocab()
        for dtype in (t.float32, t.bfloat16, t.int64):
            assert v.select_rows(t.zeros(N, 3, dtype=dtype)).dtype == dtype

    def test_select_rows_rejects_wrong_row_count(self):
        v = _vocab()
        with pytest.raises(FeatureVocabError, match="n_sae_features"):
            v.select_rows(t.zeros(N + 1, 2))

    def test_select_rows_supports_rank_3(self):
        v = _vocab((0, 3))
        full = t.randn(N, 2, 3)
        assert v.select_rows(full).shape == (2, 2, 3)


# ===========================================================================
# 7. canonical_dict / fingerprint / metadata_dict
# ===========================================================================


class TestSerialization:
    def test_canonical_dict_shape(self):
        v = _vocab((0, 3, 7, 11))
        assert v.canonical_dict() == {
            "n_sae_features": N,
            "original_ids": [0, 3, 7, 11],
        }

    def test_fingerprint_is_deterministic(self):
        assert _vocab().fingerprint() == _vocab().fingerprint()

    def test_fingerprint_is_64_hex_chars(self):
        fp = _vocab().fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_ids_give_a_different_fingerprint(self):
        assert _vocab((0, 3)).fingerprint() != _vocab((0, 7)).fingerprint()

    def test_SAME_IDS_but_different_n_sae_give_DIFFERENT_fingerprints(self):
        """Spec §4.9.3 — the silent-wrong-model bug this guards.

        {whitelist, ids:[1,2,3]} on a 16k SAE and a 65k SAE produce the SAME
        bl_/wl_ leaf hash (which covers only mode+ids), the same ids, and the same
        V=3 table shape. Only n_sae_features distinguishes them.
        """
        a = FeatureVocab(n_sae_features=16, original_ids=(1, 2, 3))
        b = FeatureVocab(n_sae_features=32, original_ids=(1, 2, 3))
        assert a != b
        assert a.fingerprint() != b.fingerprint()

    def test_metadata_dict_is_small_and_complete(self):
        """meta carries a FINGERPRINT, not the id list — the buffers are the truth."""
        v = _vocab((0, 3, 7, 11))
        md = v.metadata_dict()
        assert md == {
            "n_sae_features": N,
            "vocab_size": 4,
            "sha256": v.fingerprint(),
        }
        assert "original_ids" not in md


# ===========================================================================
# 8. The fingerprint's WIRE FORMAT is a compatibility surface
# ===========================================================================


class TestFingerprintWireFormat:
    """``fingerprint()`` is written into ``meta_*.json`` and compared byte-for-byte on
    resume (``assert_resume_feature_vocab_matches``).

    That makes ``canonical_dict()``'s key NAMES, the ``sort_keys=True`` ordering and the
    ``(",", ":")`` separators a persisted wire format, not an implementation detail. If
    any of them drifts, every already-written checkpoint's recorded fingerprint stops
    matching a freshly computed one and VALID resumes start being rejected — a
    self-inflicted outage that no other test in this file would notice, because they all
    recompute both sides with the same (new) code.

    Pin the exact bytes.
    """

    GOLDEN = "01b80fefaba2ccfff9aa4cfe237015352fff70b0faf7d95d8f32f4fa53827665"

    def test_fingerprint_matches_a_pinned_golden_digest(self):
        v = FeatureVocab(n_sae_features=16, original_ids=(0, 3, 7, 11))
        assert v.fingerprint() == self.GOLDEN, (
            "the fingerprint's wire format changed — every existing checkpoint's "
            "meta_*.json now records a stale digest and will be rejected on resume"
        )

    def test_the_golden_digest_is_the_sha256_of_the_documented_payload(self):
        """Spell the payload out, so a drift report says WHAT changed, not just 'hash'."""
        v = FeatureVocab(n_sae_features=16, original_ids=(0, 3, 7, 11))
        payload = '{"n_sae_features":16,"original_ids":[0,3,7,11]}'
        assert json.dumps(
            v.canonical_dict(), sort_keys=True, separators=(",", ":")
        ) == payload
        assert hashlib.sha256(payload.encode()).hexdigest() == self.GOLDEN

    def test_source_ORDER_and_duplicates_cannot_change_the_digest(self):
        """Canonicalisation happens before hashing, so a caller that hands the ids in a
        different order must not produce a different checkpoint fingerprint."""
        shuffled = FeatureVocab(n_sae_features=16, original_ids=(11, 3, 0, 7, 3, 11))
        assert shuffled.fingerprint() == self.GOLDEN


# ===========================================================================
# 9. Adversarial id translation
# ===========================================================================


class TestAdversarialIdTranslation:
    def test_to_original_of_an_OOV_compact_id_RAISES_rather_than_WRAPPING(self):
        """THE composition trap, and the reason ``to_original`` bounds-checks at all.

        Torch indexes ``-1`` as the LAST row — so an unguarded ``new_to_old_tensor()[-1]``
        would silently hand back the vocab's LARGEST original id and the round trip would
        look successful. The guard is what turns that into an error.

        ``to_compact`` can no longer *produce* the ``-1``; it raises first. But
        ``old_to_new_tensor()`` is public and is still documented as ``-1`` where
        out-of-vocab, so a caller can still route one into ``to_original``. Both halves
        are asserted: the producer refuses, and the consumer guards anyway.
        """
        v = _vocab((0, 3, 7, 11))

        # The producer refuses: there is no way to get -1 out of `to_compact` at all.
        with pytest.raises(FeatureVocabError):
            v.to_compact(t.tensor([5], dtype=t.int64))

        # The raw map still carries the sentinel, and a caller can still reach it.
        compact = v.old_to_new_tensor()[t.tensor([5])]
        assert compact.tolist() == [-1]

        # What the unguarded gather WOULD have returned — a plausible, wrong id:
        assert int(v.new_to_old_tensor()[-1]) == 11

        with pytest.raises(FeatureVocabError, match="out of range"):
            v.to_original(compact)

    def test_duplicate_ids_all_map_to_the_SAME_compact_row(self):
        """A segment may legitimately repeat a feature id; the map is elementwise."""
        v = _vocab((0, 3, 7, 11))
        assert v.to_compact(t.tensor([7, 7, 7], dtype=t.int64)).tolist() == [2, 2, 2]

    def test_UNSORTED_input_maps_by_RANK_not_by_position(self):
        """compact id == rank of the ORIGINAL id in the vocab. It is a property of the
        id, never of where the id happened to sit in the caller's tensor."""
        v = _vocab((0, 3, 7, 11))
        assert v.to_compact(t.tensor([11, 0, 7, 3], dtype=t.int64)).tolist() == [3, 0, 2, 1]

    def test_id_exactly_n_sae_is_rejected_the_off_by_one_boundary(self):
        v = _vocab()
        with pytest.raises(FeatureVocabError, match="out of range"):
            v.to_compact(t.tensor([0, N], dtype=t.int64))

    def test_the_LARGEST_LEGAL_id_is_IN_RANGE_whether_or_not_it_is_IN_VOCAB(self):
        """``n_sae - 1`` is a legal SAE id. It must compact when the vocab holds it, and
        must raise the OUT-OF-VOCAB error when it does not — never be swept up by the
        RANGE check, which is a different failure with a different fix.

        Stronger than the sentinel version this replaces: that one could only observe
        *that* the boundary id was missing, not WHICH of the two errors it earned.
        """
        inside = _vocab((0, N - 1))
        assert inside.to_compact(t.tensor([N - 1], dtype=t.int64)).tolist() == [1]

        outside = _vocab((0, 3))
        with pytest.raises(FeatureVocabError) as exc:
            outside.to_compact(t.tensor([N - 1], dtype=t.int64))
        msg = str(exc.value)
        assert f"[{N - 1}]" in msg, "the out-of-vocab error must name the offending id"
        assert "out of range" not in msg, (
            f"id {N - 1} IS in range [0, {N}) — it is merely out of VOCAB. Reporting it as "
            "out-of-range would send the operator to fix the wrong thing."
        )

    def test_a_MIXED_tensor_maps_each_element_independently(self):
        """The MAPPING is elementwise; the VALIDATION is all-or-nothing.

        Every element is the rank of its OWN id, wherever it sits. But one excluded id
        anywhere is fatal for the WHOLE tensor — the only safe reading, since there is no
        representable value for the invalid slots.
        """
        v = _vocab((0, 3, 7, 11))
        assert v.to_compact(t.tensor([[11, 0], [7, 3]], dtype=t.int64)).tolist() == [
            [3, 0],
            [2, 1],
        ]

        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([[0, 1], [7, 15]], dtype=t.int64))
        assert "[1, 15]" in str(exc.value), f"offenders misreported: {exc.value}"

    def test_to_compact_upcasts_int32_ids_to_int64(self):
        """Producers are free to emit int32; embedding indices must come back int64."""
        v = _vocab((0, 3, 7, 11))
        got = v.to_compact(t.tensor([0, 3], dtype=t.int32))
        assert got.dtype == t.int64
        assert got.tolist() == [0, 1]

    def test_to_compact_preserves_rank_3_shape(self):
        v = _vocab((0, 3, 7, 11))
        assert v.to_compact(t.zeros(2, 3, 4, dtype=t.int64)).shape == (2, 3, 4)


class TestSaeIdZeroBoundary:
    """SAE id 0 is special: every producer pads with a literal ``0``, and compact row 0
    is where padding lands. Whether id 0 is IN the vocab changes what row 0 *means*."""

    def test_when_id_0_IS_in_the_vocab_compact_row_0_IS_original_id_0(self):
        v = _vocab((0, 3, 7))
        assert int(v.to_compact(t.tensor([0]))[0]) == 0
        assert int(v.to_original(t.tensor([0]))[0]) == 0

    def test_when_id_0_is_NOT_in_the_vocab_compact_row_0_is_a_DIFFERENT_feature(self):
        """The whitelist case, and the mechanism behind the documented FP8 exception:
        padding still embeds compact row 0, but row 0 now carries original id 3.

        The asymmetry IS the point, and it survives the sentinel's removal intact:
        compact row 0 exists (it is id 3), while ORIGINAL id 0 does not exist in this
        vocab at all — so ``to_original(0)`` answers and ``to_compact(0)`` raises.
        """
        v = _vocab((3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([0]))
        assert "[0]" in str(exc.value), "id 0 must be reported as the out-of-vocab id"
        assert int(v.to_original(t.tensor([0]))[0]) == 3, "row 0 exists; it is id 3"


class TestExtremeShapes:
    def test_n_sae_of_ONE_is_the_identity_and_normalizes_to_None(self):
        m = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0,))
        assert FeatureVocab.from_mask(m, 1) is None

    def test_blacklisting_the_ONLY_feature_raises(self):
        m = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0,))
        with pytest.raises(FeatureVocabError, match="zero features"):
            FeatureVocab.from_mask(m, 1)

    def test_V_of_ONE_still_produces_a_coherent_map(self):
        """The most extreme compaction the encoder can be asked to build."""
        v = FeatureVocab(n_sae_features=N, original_ids=(9,))
        assert v.size == 1 and not v.is_identity
        o2n = v.old_to_new_tensor()
        assert o2n.tolist() == [-1] * 9 + [0] + [-1] * (N - 10)
        assert v.new_to_old_tensor().tolist() == [9]
        assert v.select_rows(t.arange(N, dtype=t.float32).unsqueeze(1)).tolist() == [[9.0]]
        assert v.to_original(v.to_compact(t.tensor([9]))).tolist() == [9]

    def test_dropping_exactly_ONE_feature_is_still_a_non_identity_vocab(self):
        """V == n_sae - 1: the near-identity boundary. It must NOT normalize away."""
        ids = tuple(i for i in range(N) if i != 5)
        v = FeatureVocab(n_sae_features=N, original_ids=ids)
        assert not v.is_identity
        assert v.size == N - 1
        with pytest.raises(FeatureVocabError, match=r"\[5\]"):
            v.to_compact(t.tensor([5]))  # 5 is the one hole in the vocab
        # ids below the hole keep their index; ids above shift down by exactly one.
        assert int(v.to_compact(t.tensor([4]))[0]) == 4
        assert int(v.to_compact(t.tensor([6]))[0]) == 5

    def test_from_mask_at_V_equals_n_sae_minus_one_does_NOT_normalize_to_None(self):
        m = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(5,))
        v = FeatureVocab.from_mask(m, N)
        assert v is not None, "only a TRUE identity may normalize to None"
        assert v.size == N - 1


# ===========================================================================
# 10. to_compact's OUT-OF-VOCAB POLICY — it RAISES, and that is the ONLY codepath
# ===========================================================================


class TestToCompactAlwaysRaisesOnOutOfVocab:
    """``-1`` is a LETHAL return value, so the public API must never hand it out — not
    silently, and not by opt-in either.

    ``FeatureEncoder._to_compact`` guards its remap with ``torch._assert_async`` for one
    reason, stated in caps in its own docstring: under ``torch.compile``,
    ``nn.Embedding(-1)`` **silently returns the LAST row**. The identically-named public
    ``FeatureVocab.to_compact`` has no such guard, and no mask with which to tell dead
    padding from a real id, so it cannot afford to return ``-1`` at all.

    Concretely: ``EncoderDecoderGPT.feature_vocab`` is deliberately exposed so the FP-head
    follow-up can reach it. That follow-up calls ``model.feature_vocab.to_compact(bag_ids)``
    on a bag holding one mask-excluded feature. A ``-1`` there indexes a compiled
    ``nn.Embedding(V, d)``, reads the LAST vocab row, and silently conditions on — and
    scores — the WRONG SAE feature, with no error anywhere.

    An opt-in ``on_missing="sentinel"`` briefly existed here. It is GONE, deliberately: an
    escape hatch for a silent failure is still a codepath, and the next caller in a hurry
    reaches for it. Raising is not the default — it is the only behaviour. See
    ``TestTheSentinelEscapeHatchCannotComeBack`` for the pins that keep it that way.
    """

    def test_an_in_range_out_of_vocab_id_RAISES(self):
        """THE finding. id 5 is a legal SAE id, but the mask excluded it."""
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError):
            v.to_compact(t.tensor([0, 5], dtype=t.int64))

    def test_the_error_NAMES_the_offending_ids(self):
        """'some id was out of vocab' is not actionable. Say WHICH."""
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError, match=r"\b5\b") as exc:
            v.to_compact(t.tensor([0, 5, 3], dtype=t.int64))
        assert "5" in str(exc.value)

    def test_the_error_names_the_REMEDY(self):
        """Fail-loud means naming the fix, not just the fault."""
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([5], dtype=t.int64))
        msg = str(exc.value)
        assert "feature_mask" in msg, "the error must point at the mask that excluded it"
        assert "dump_feature_mask" in msg, (
            "the error must name the command that recovers the mask — 'drop the bad ids' "
            "is not actionable unless the operator can find out which ids those are"
        )

    def test_the_error_REFUSES_to_offer_a_sentinel_ESCAPE_HATCH(self):
        """The message must not send the reader hunting for an opt-in that no longer
        exists — and must say, in as many words, that there deliberately is none.

        This message is where an operator meets this decision at 3am, and 'is there a flag
        for this?' is the very first thing they will ask. Answer it here, or they will go
        and add the flag back.
        """
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([5], dtype=t.int64))
        msg = str(exc.value)
        assert "on_missing" not in msg, (
            "the error must not advertise an `on_missing` opt-in — the parameter is gone"
        )
        assert "no option" in msg.lower(), (
            "the error must state that there is deliberately NO way to get the -1 back"
        )

    def test_the_error_explains_WHY_minus_one_is_not_merely_returned(self):
        """The reason is not obvious and is the whole point: -1 is not a safe sentinel."""
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([5], dtype=t.int64))
        msg = str(exc.value)
        assert "compile" in msg.lower()
        assert "last row" in msg.lower()

    def test_the_error_names_EVERY_distinct_offender_not_just_the_first(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.tensor([5, 9, 5, 12], dtype=t.int64))
        msg = str(exc.value)
        for offender in ("5", "9", "12"):
            assert offender in msg, f"offending id {offender} is not named"

    def test_a_LARGE_number_of_offenders_is_truncated_but_COUNTED(self):
        """The message must stay readable without lying about the scale of the problem."""
        v = FeatureVocab(n_sae_features=1024, original_ids=(0, 1))
        with pytest.raises(FeatureVocabError) as exc:
            v.to_compact(t.arange(2, 900, dtype=t.int64))
        msg = str(exc.value)
        assert "898" in msg, "the TOTAL number of offending ids must be stated"
        assert len(msg) < 2000, "the message must not dump 898 ids"

    # -- interaction with the pre-existing RANGE check ----------------------

    def test_an_OUT_OF_RANGE_id_raises_the_RANGE_error_NOT_the_out_of_vocab_error(self):
        """The two failures are different, and the message must not conflate them.

        An id ``>= n_sae`` (or ``< 0``) is not an SAE feature id AT ALL: no mask excluded
        it, and no mask change will admit it — the caller's ids are simply wrong. Telling
        that operator "these were excluded by the feature_mask, drop them from your input"
        would send them to widen a mask that was never the problem.
        """
        v = _vocab()
        for bad in ([0, N], [0, -1]):
            with pytest.raises(FeatureVocabError) as exc:
                v.to_compact(t.tensor(bad, dtype=t.int64))
            msg = str(exc.value)
            assert "out of range" in msg
            assert "feature_mask" not in msg, (
                f"{bad} is not an out-of-VOCAB failure — no feature_mask could have "
                "admitted it, so the mask must not be named as the remedy"
            )

    def test_an_EMPTY_tensor_returns_an_empty_int64_tensor_and_never_raises(self):
        """The empty input takes an EARLY RETURN that skips both the range check and the
        out-of-vocab check — the one codepath in ``to_compact`` that validates nothing.

        So pin what it returns rather than merely that it survives: the shape is preserved
        (a shaped empty stays shaped) and the dtype is normalised to int64 even from a
        non-int64 input, exactly as on the strict path.
        """
        v = _vocab()
        assert v.to_compact(t.tensor([], dtype=t.int64)).numel() == 0

        for empty in (t.zeros(0, dtype=t.int32), t.zeros(0, 4, dtype=t.int64)):
            got = v.to_compact(empty)
            assert got.shape == empty.shape
            assert got.dtype == t.int64

    # -- shape / dtype on the (only) path -----------------------------------

    def test_the_strict_path_preserves_shape_and_dtype_on_the_happy_path(self):
        v = _vocab((0, 3, 7, 11))
        got = v.to_compact(t.tensor([[0, 3], [7, 11]], dtype=t.int32))
        assert got.shape == (2, 2)
        assert got.dtype == t.int64
        assert got.tolist() == [[0, 1], [2, 3]]

    def test_a_single_out_of_vocab_id_hidden_in_a_LARGE_in_vocab_tensor_still_raises(self):
        """The FP-head scenario verbatim: one mask-excluded feature in an otherwise valid
        bag. Nothing about the shape or the other 99 ids makes it visible."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        ids = t.full((10, 10), 7, dtype=t.int64)
        ids[6, 2] = 5  # the one excluded feature
        with pytest.raises(FeatureVocabError, match=r"\b5\b"):
            v.to_compact(ids)


# ===========================================================================
# 11. The escape hatch cannot come back
# ===========================================================================


class TestTheSentinelEscapeHatchCannotComeBack:
    """``to_compact`` briefly took ``on_missing="sentinel"`` — an opt-in that returned the
    ``-1``. It was deleted, deliberately: an escape hatch for a silent failure is still a
    codepath, and the next caller in a hurry reaches for it.

    Deleting the parameter is not self-enforcing. The tests that PINNED it were deleted
    along with it, so nothing else in this file would notice it coming back. These are the
    pins that replace them: they fail if anyone restores the keyword, the ``OnMissing``
    alias, the policy tuple, or any other route to a negative return value.
    """

    def test_to_compact_takes_EXACTLY_the_ids_and_nothing_else(self):
        """Pin the SIGNATURE, not just the behaviour — 'add the flag back' is exactly the
        kind of edit that gets made while trying to get some *other* test to pass."""
        params = inspect.signature(FeatureVocab.to_compact).parameters
        assert list(params) == ["self", "ids"], (
            f"to_compact's signature grew a parameter: {list(params)}. If that parameter "
            "is an out-of-vocab policy, read this class's docstring before keeping it."
        )
        assert not any(
            p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            for p in params.values()
        ), "**kwargs would silently SWALLOW an on_missing= instead of rejecting it"

    @pytest.mark.parametrize("ids", [[0, 3], [0, 5], []])
    @pytest.mark.parametrize("value", ["sentinel", "raise", "ignore", None, True])
    def test_passing_on_missing_AT_ALL_is_a_TypeError(self, value, ids):
        """Any value, and any input: fully in-vocab, out-of-vocab, and empty.

        The deleted parameter was validated UP FRONT so that a typo'd policy could not lie
        dormant until the first out-of-vocab id in production. Removing it keeps that
        property and strengthens it: this is now a TypeError raised from the signature, so
        it fires before a single tensor element is looked at — including on the empty
        tensor, whose early return validates nothing at all.
        """
        v = _vocab((0, 3, 7, 11))
        with pytest.raises(TypeError, match="on_missing"):
            v.to_compact(t.tensor(ids, dtype=t.int64), on_missing=value)  # type: ignore[call-arg]  # ty: ignore[unknown-argument]

    def test_a_SECOND_POSITIONAL_argument_is_a_TypeError(self):
        """The one claim the deleted keyword-only test was REALLY making, and it still
        holds — so it is kept rather than dropped with the parameter that occasioned it.

        The encoder's same-named PRIVATE method genuinely does take a mask second
        (``FeatureEncoder._to_compact(ids, mask)``). Anyone copying that call shape onto
        the public method must get a loud error, not a quietly-ignored second argument —
        because a mask-unaware ``to_compact`` given a mask would validate the padding
        slots it was being handed the mask in order to skip.
        """
        v = _vocab((0, 3, 7, 11))
        mask = t.ones(1, dtype=t.bool)
        with pytest.raises(TypeError):
            v.to_compact(t.tensor([0], dtype=t.int64), mask)  # type: ignore[call-arg]  # ty: ignore[too-many-positional-arguments]

    def test_the_OnMissing_alias_and_the_policy_tuple_are_GONE_from_the_module(self):
        """The keyword was the door; these were the doorframe. A restored ``OnMissing``
        alias is the first commit of the change that restores the parameter."""
        for name in ("OnMissing", "_ON_MISSING_POLICIES"):
            assert not hasattr(feature_vocab_module, name), (
                f"{name} is back in steering.conditional.feature_vocab — the -1 sentinel "
                "escape hatch was removed deliberately; see FeatureVocab.to_compact"
            )

    def test_to_compact_NEVER_returns_a_negative_value_for_ANY_input_it_accepts(self):
        """The invariant the whole tightening buys, over the ENTIRE legal input domain:
        every id in ``[0, n_sae)``, against vocabs that hold it and vocabs that do not.

        A compact id is an index into ``nn.Embedding(V, d)``, and under ``torch.compile``
        a negative index does not raise — it reads from the END. So the contract has
        exactly two outcomes: every value returned is a REAL row of that table, and every
        id that cannot be mapped is an exception. There is no third one.
        """
        for v in (_vocab((0, 3, 7, 11)), _vocab((3,)), _vocab(tuple(range(N - 1)))):
            for i in range(N):  # the whole legal domain [0, n_sae)
                try:
                    got = v.to_compact(t.tensor([[i, i]], dtype=t.int64))
                except FeatureVocabError:
                    continue  # the only OTHER permitted outcome
                assert int(got.min()) >= 0, f"to_compact({i}) returned a negative row"
                assert int(got.max()) < v.size, f"to_compact({i}) returned a row >= V={v.size}"
