"""Checkpoint guards for FeatureVocab (#695 PR A).

Two pure functions, both of which exist to turn a SILENT wrong-model load into a
loud, actionable one:

* ``feature_vocab_from_state_dict`` — makes a checkpoint SELF-DESCRIBING. Today
  ``checkpoint_manager.py:523`` infers ``n_sae_features`` from
  ``encoder.feature_embed.weight.shape[0]``; once the table is compacted that row
  count is ``V``, not ``n_sae``. Every eval consumer that reads
  ``model.encoder.config.n_sae_features`` would silently get the wrong number.

* ``assert_checkpoint_vocab_compatible`` — runs BEFORE ``load_state_dict`` so the
  operator gets a real message instead of PyTorch's bare ``size mismatch``.

* ``assert_resume_feature_vocab_matches`` — the meta-side guard. It must compare
  ``n_sae_features`` as well as the ids; comparing ids alone is NOT sufficient (see
  ``test_same_ids_different_n_sae_is_the_silent_wrong_model_bug``).
"""

from __future__ import annotations

import re

import pytest
import torch as t

from errors import FeatureVocabError
from steering.conditional.encoder_decoder.encoder import FeatureEncoder, FeatureEncoderConfig
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
)
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
    assert_checkpoint_vocab_compatible,
    assert_resume_feature_vocab_matches,
    feature_vocab_from_state_dict,
)

N = 16
CKPT = "/fake/conditional_encdec_checkpoints/d12/s128/t2048.9/bl_deadbeefdeadbeef"


def _legacy_state_dict(n_sae: int = N, dim: int = 4) -> dict[str, t.Tensor]:
    """A pre-#695 checkpoint: full-size tables, NO vocab buffers."""
    return {
        FEATURE_EMBED_KEY: t.zeros(n_sae, dim),
        "encoder.magnitude_embed.weight": t.zeros(2, dim),
    }


def _compact_state_dict(vocab: FeatureVocab, dim: int = 4) -> dict[str, t.Tensor]:
    """A post-#695 compacted checkpoint: V-row tables + the two vocab buffers."""
    return {
        FEATURE_EMBED_KEY: t.zeros(vocab.size, dim),
        "encoder.magnitude_embed.weight": t.zeros(2, dim),
        VOCAB_IDS_KEY: t.tensor(vocab.original_ids, dtype=t.int64),
        VOCAB_N_SAE_KEY: t.tensor(vocab.n_sae_features, dtype=t.int64),
    }


def _identity_state_dict(n_sae: int = N, dim: int = 4) -> dict[str, t.Tensor]:
    """A checkpoint whose vocab buffers describe the IDENTITY (V == n_sae).

    No writer in this repo can produce one — ``FeatureVocab.from_mask`` normalizes an
    identity to ``None``, so the model that would have written the buffers never
    registers them. It is constructed here precisely because it is the state the two
    constructors used to DISAGREE about.
    """
    return {
        FEATURE_EMBED_KEY: t.zeros(n_sae, dim),
        "encoder.magnitude_embed.weight": t.zeros(2, dim),
        VOCAB_IDS_KEY: t.arange(n_sae, dtype=t.int64),
        VOCAB_N_SAE_KEY: t.tensor(n_sae, dtype=t.int64),
    }


def _real_uncompacted_encoder(n_sae: int = N) -> FeatureEncoder:
    """A REAL, uncompacted FeatureEncoder — the model an identity vocab resolves to."""
    cfg = FeatureEncoderConfig(
        n_sae_features=n_sae,
        num_buckets=4,
        n_embd=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )
    with t.device("meta"):
        enc = FeatureEncoder(cfg, feature_vocab=None)
    enc.to_empty(device="cpu")
    enc.init_weights()
    return enc


# ===========================================================================
# feature_vocab_from_state_dict — self-describing checkpoints
# ===========================================================================


class TestFromStateDict:
    def test_legacy_checkpoint_yields_identity_and_the_shape_inferred_width(self):
        """Requirement 5: existing checkpoints, which carry no vocab, load as identity.

        For an identity vocab the OLD shape-inference is still CORRECT
        (feature_embed.shape[0] == n_sae_features), so legacy loads are unchanged.
        """
        n_sae, vocab = feature_vocab_from_state_dict(_legacy_state_dict(n_sae=N))
        assert n_sae == N
        assert vocab is None

    def test_compacted_checkpoint_is_reconstructed_from_the_FILE_ALONE(self):
        """Requirement 1: no training config, no sidecar. The checkpoint describes itself."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        n_sae, got = feature_vocab_from_state_dict(_compact_state_dict(v))
        assert got == v
        assert n_sae == N, "n_sae must be the ORIGINAL width, not V"

    def test_compacted_n_sae_is_NOT_the_table_row_count(self):
        """The whole trap: V != n_sae once compacted."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        sd = _compact_state_dict(v)
        assert sd[FEATURE_EMBED_KEY].shape[0] == 4  # V
        n_sae, _ = feature_vocab_from_state_dict(sd)
        assert n_sae == N == 16  # NOT 4

    def test_missing_feature_embed_raises(self):
        with pytest.raises(FeatureVocabError, match="encoder-decoder checkpoint"):
            feature_vocab_from_state_dict({"transformer.wte.weight": t.zeros(4, 4)})

    def test_vocab_ids_without_n_sae_is_a_corrupt_checkpoint(self):
        sd = _legacy_state_dict()
        sd[VOCAB_IDS_KEY] = t.tensor([0, 3], dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="corrupt|both|missing"):
            feature_vocab_from_state_dict(sd)

    def test_n_sae_without_vocab_ids_is_a_corrupt_checkpoint(self):
        sd = _legacy_state_dict()
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="corrupt|both|missing"):
            feature_vocab_from_state_dict(sd)

    def test_row_count_disagreeing_with_vocab_size_raises(self):
        """feature_embed has V rows by definition. If it doesn't, the file is corrupt."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        sd = _compact_state_dict(v)
        sd[FEATURE_EMBED_KEY] = t.zeros(99, 4)  # wrong row count
        with pytest.raises(FeatureVocabError, match="rows"):
            feature_vocab_from_state_dict(sd)

    def test_UNSORTED_vocab_buffer_raises_instead_of_being_silently_re_sorted(self):
        """A silent-corruption hole, caught in review.

        ``FeatureVocab.__post_init__`` CANONICALISES (sorts) its ids. So an unsorted
        ``feature_vocab_ids`` buffer would be silently re-sorted, and compact row j
        would then be reinterpreted as a DIFFERENT feature than the weights encode —
        with no error anywhere.

        The buffer's ORDER *is* the row mapping: row j means original_ids[j]. It must
        be taken literally, never re-derived. Unreachable from the production encoder
        (which writes sorted ids), which is exactly why it needs a test.
        """
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(3, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([7, 3, 0], dtype=t.int64)  # NOT sorted
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="(?i)sorted|ascending|order"):
            feature_vocab_from_state_dict(sd)

    def test_duplicate_ids_in_the_vocab_buffer_raise(self):
        """__post_init__ de-duplicates too, which would silently shrink V below the
        table's row count."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(3, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([0, 3, 3], dtype=t.int64)  # duplicate
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError):
            feature_vocab_from_state_dict(sd)

    def test_ids_out_of_range_are_rejected(self):
        """An id >= n_sae is not merely out-of-vocab, it is not an SAE feature id at all, so a
        checkpoint whose vocab buffer contains one is rejected at load."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([0, N], dtype=t.int64)  # N is out of [0, N)
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="is out of range"):
            feature_vocab_from_state_dict(sd)

    def test_an_EMPTY_ids_buffer_is_rejected(self):
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(0, 4)
        sd[VOCAB_IDS_KEY] = t.zeros(0, dtype=t.int64)
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="feature vocab is empty"):
            feature_vocab_from_state_dict(sd)

    def test_a_NON_INTEGER_ids_buffer_is_rejected_rather_than_truncated(self):
        """`int(3.7)` is 3, so a float buffer would TRUNCATE into a different vocabulary
        that looks entirely plausible: compact row 1 would mean feature 3 while the weights
        encode a different row, with nothing downstream to catch it. The module already
        rejects unsorted and duplicate ids precisely because they are unreachable from the
        production encoder — a non-integral dtype is unreachable by the same argument, and
        must not be the one corruption that is quietly coerced."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([0.0, 3.7], dtype=t.float32)
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="must be an integer tensor"):
            feature_vocab_from_state_dict(sd)

    def test_a_BOOL_ids_buffer_is_rejected_rather_than_coerced_to_0_1(self):
        """`int(True)` is 1, so a bool buffer would COERCE into a different vocabulary that
        looks entirely plausible: compact row 0 would mean feature 0 or 1 depending on the
        flag, with nothing downstream to catch it. `torch.bool` is neither floating-point
        nor complex, so it survives those two checks and needs its own explicit guard —
        exactly the trap this test pins."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([False, True], dtype=t.bool)
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="must be an integer tensor"):
            feature_vocab_from_state_dict(sd)


# ===========================================================================
# The IDENTITY vocab — the two constructors must AGREE
# ===========================================================================


class TestIdentityVocabIsRefused:
    """A checkpoint carrying IDENTITY vocab buffers is an ANOMALY, and it now RAISES.

    THE INVARIANT: a non-``None`` ``FeatureVocab`` ALWAYS means "genuinely compacted"
    (``V < n_sae``). Neither constructor may ever return an identity — but they refuse it
    DIFFERENTLY, and the asymmetry is the whole point:

    * ``from_mask`` NORMALIZES an identity to ``None``. An all-allowing mask is a legitimate
      config and must build a full-width, byte-identical encoder (invariant I5).
    * ``feature_vocab_from_state_dict`` RAISES. Because ``from_mask`` normalizes, the model
      that would have registered those buffers never registers them — so NO writer in this
      repo can emit such a checkpoint. It came from a writer we do not understand.

    This class used to assert the opposite: that the state-dict constructor warned,
    normalized to ``None``, and DELETED the two buffers from the caller's state dict so the
    uncompacted model it implied could still strict-load. That was leniency in the one place
    it is least affordable — it silently reinterpreted a checkpoint from an unknown source —
    and it forced this function to MUTATE a dict its callers read as a pure query. Removing
    it deleted the mutation too.
    """

    def test_identity_buffers_RAISE(self):
        with pytest.raises(FeatureVocabError, match="(?i)identity") as exc:
            feature_vocab_from_state_dict(_identity_state_dict())
        msg = str(exc.value)
        assert "no writer" in msg.lower(), "must say WHY this is anomalous, not just that"
        assert "strip" in msg.lower(), "must name the remedy"

    def test_the_function_is_a_PURE_QUERY_and_does_not_mutate_the_state_dict(self):
        """The mutation existed ONLY to serve the identity normalization. It is gone, and
        this pins that it stays gone: a reader that silently deletes the caller's keys is a
        trap regardless of how well-intentioned."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        sd = _compact_state_dict(v)
        before = dict(sd)
        feature_vocab_from_state_dict(sd)
        assert set(sd) == set(before), "feature_vocab_from_state_dict must not mutate"
        assert all(t.equal(sd[k], before[k]) for k in before)

    def test_the_TWO_CONSTRUCTORS_BOTH_REFUSE_to_return_an_identity(self):
        """The invariant, stated as the thing that actually matters: neither can hand back
        an identity vocab. One returns None, one raises — both uphold it."""
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(N)))
        assert FeatureVocab.from_mask(mask, N) is None  # legitimate config -> uncompacted
        with pytest.raises(FeatureVocabError):  # anomalous checkpoint -> refuse
            feature_vocab_from_state_dict(_identity_state_dict())

    def test_a_NON_None_vocab_ALWAYS_means_genuinely_compacted(self):
        """The invariant the normalization restores, asserted as such."""
        _, vocab = feature_vocab_from_state_dict(
            _compact_state_dict(FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11)))
        )
        assert vocab is not None
        assert not vocab.is_identity
        assert vocab.size < vocab.n_sae_features

    def test_a_GENUINELY_compacted_checkpoint_is_untouched(self):
        """Guard against over-normalizing: only a TRUE identity may collapse to None."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        sd = _compact_state_dict(v)
        n_sae, got = feature_vocab_from_state_dict(sd)
        assert got == v
        assert n_sae == N
        assert VOCAB_IDS_KEY in sd, "a real compaction keeps its buffers"
        assert VOCAB_N_SAE_KEY in sd

    def test_the_NEAR_identity_V_equals_n_sae_minus_one_does_NOT_normalize(self):
        """The boundary. Dropping exactly one feature IS a compaction."""
        v = FeatureVocab(n_sae_features=N, original_ids=tuple(i for i in range(N) if i != 5))
        _, got = feature_vocab_from_state_dict(_compact_state_dict(v))
        assert got == v
        assert got is not None and got.size == N - 1

    # -- the round trip ----------------------------------------------------

    def test_the_REMEDY_the_error_names_actually_works(self):
        """The error tells the operator to STRIP the buffers. Take it at its word: do that,
        and the checkpoint loads into a real uncompacted encoder through a REAL strict load
        (only ``load_state_dict(strict=True)`` decides what an 'unexpected key' is)."""
        enc = _real_uncompacted_encoder(n_sae=N)
        sd = dict(enc.state_dict())
        sd[VOCAB_IDS_KEY.removeprefix("encoder.")] = t.arange(N, dtype=t.int64)
        sd[VOCAB_N_SAE_KEY.removeprefix("encoder.")] = t.tensor(N, dtype=t.int64)
        prefixed = {f"encoder.{k}": v for k, v in sd.items()}

        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            feature_vocab_from_state_dict(prefixed)

        # The remedy: strip them. Now it is an ordinary uncompacted checkpoint.
        del prefixed[VOCAB_IDS_KEY]
        del prefixed[VOCAB_N_SAE_KEY]
        n_sae, vocab = feature_vocab_from_state_dict(prefixed)
        assert vocab is None and n_sae == N
        cleaned = {k.removeprefix("encoder."): v for k, v in prefixed.items()}
        enc.load_state_dict(cleaned, strict=True)

    def test_refusal_does_NOT_swallow_a_corrupt_row_count(self):
        """The identity path must run AFTER the integrity checks, not instead of them."""
        sd = _identity_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(99, 4)  # not N rows ⇒ corrupt
        with pytest.raises(FeatureVocabError, match="rows"):
            feature_vocab_from_state_dict(sd)

    def test_refusal_does_NOT_swallow_an_UNSORTED_identity_permutation(self):
        """``[1, 0, 2, 3, ...]`` is a permutation of ``range(n_sae)`` — it would become an
        identity AFTER __post_init__ sorts it. But the buffer's ORDER *is* the row mapping,
        so a permuted buffer means the weights are permuted too, and silently calling it
        'identity' would reinterpret every row. It must still be rejected as unsorted."""
        sd = _identity_state_dict()
        ids = list(range(N))
        ids[0], ids[1] = ids[1], ids[0]
        sd[VOCAB_IDS_KEY] = t.tensor(ids, dtype=t.int64)
        with pytest.raises(FeatureVocabError, match="(?i)sorted|ascending|order"):
            feature_vocab_from_state_dict(sd)

    # -- the two downstream consequences named in the finding ---------------

    def test_the_compat_guard_also_refuses_an_identity_checkpoint(self):
        """It forwards to feature_vocab_from_state_dict, so it inherits the refusal — and it
        must, or the two entry points would disagree about the same file."""
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            assert_checkpoint_vocab_compatible(None, _identity_state_dict(), checkpoint_dir=CKPT)

    def test_a_genuinely_compacted_checkpoint_STILL_trips_that_guard(self):
        """The guard must keep working — the fix narrows it, it does not disable it."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        with pytest.raises(FeatureVocabError, match="COMPACTED"):
            assert_checkpoint_vocab_compatible(None, _compact_state_dict(v), checkpoint_dir=CKPT)


# ===========================================================================
# Buffer RANK / SHAPE corruption — a FeatureVocabError, never a bare torch error
# ===========================================================================


class TestBufferRankAndShape:
    """Both callers document and DEPEND ON the ``FeatureVocabError`` contract:
    ``dump_feature_mask``'s docstring promises it, and ``build_encdec_model`` raises it for
    the meta-mirror mismatch immediately after this call.

    The dtype / sorted / duplicate / row-count checks all honour that contract. Rank and
    shape did not: they escaped as PyTorch's bare, unactionable errors.
    """

    def test_a_MULTI_ELEMENT_n_sae_buffer_raises_FeatureVocabError(self):
        """``int(buf.item())`` on a >1-element tensor raises PyTorch's bare
        ``RuntimeError: a Tensor with 2 elements cannot be converted to Scalar``."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([0, 3], dtype=t.int64)
        sd[VOCAB_N_SAE_KEY] = t.tensor([N, N], dtype=t.int64)  # rank-1, 2 elements
        with pytest.raises(FeatureVocabError) as exc:
            feature_vocab_from_state_dict(sd)
        msg = str(exc.value)
        assert VOCAB_N_SAE_KEY in msg, "the error must name the corrupt buffer"
        assert "2" in msg, "the error must name the offending shape/element count"

    def test_an_EMPTY_n_sae_buffer_raises_FeatureVocabError(self):
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([0, 3], dtype=t.int64)
        sd[VOCAB_N_SAE_KEY] = t.zeros(0, dtype=t.int64)
        with pytest.raises(FeatureVocabError):
            feature_vocab_from_state_dict(sd)

    def test_a_0_DIM_ids_buffer_raises_FeatureVocabError(self):
        """``tuple(int(i) for i in buf.tolist())`` on a 0-dim tensor raises PyTorch's bare
        ``TypeError: 'int' object is not iterable``."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(1, 4)
        sd[VOCAB_IDS_KEY] = t.tensor(3, dtype=t.int64)  # 0-dim, NOT a list of ids
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError) as exc:
            feature_vocab_from_state_dict(sd)
        assert VOCAB_IDS_KEY in str(exc.value)

    def test_a_2D_ids_buffer_raises_FeatureVocabError(self):
        """``.tolist()`` yields NESTED lists, and ``int([0, 3])`` is a bare TypeError."""
        sd = _legacy_state_dict()
        sd[FEATURE_EMBED_KEY] = t.zeros(4, 4)
        sd[VOCAB_IDS_KEY] = t.tensor([[0, 3], [7, 11]], dtype=t.int64)
        sd[VOCAB_N_SAE_KEY] = t.tensor(N, dtype=t.int64)
        with pytest.raises(FeatureVocabError) as exc:
            feature_vocab_from_state_dict(sd)
        msg = str(exc.value)
        assert VOCAB_IDS_KEY in msg
        assert "(2, 2)" in msg, "the error must name the offending shape"

    def test_the_rank_errors_are_NOT_bare_torch_errors(self):
        """The whole point of the finding. ``FeatureVocabError`` is a ``NanoGPTError``,
        which is a plain ``Exception`` — it is NOT a ``RuntimeError`` and NOT a
        ``TypeError``, so an ``except FeatureVocabError`` handler could never have caught
        what PyTorch used to throw here."""
        assert not issubclass(FeatureVocabError, RuntimeError)
        assert not issubclass(FeatureVocabError, TypeError)

        for bad_ids, bad_n_sae in (
            (t.tensor(3, dtype=t.int64), t.tensor(N, dtype=t.int64)),
            (t.tensor([[0, 3]], dtype=t.int64), t.tensor(N, dtype=t.int64)),
            (t.tensor([0, 3], dtype=t.int64), t.tensor([N, N], dtype=t.int64)),
        ):
            sd = _legacy_state_dict()
            sd[FEATURE_EMBED_KEY] = t.zeros(2, 4)
            sd[VOCAB_IDS_KEY] = bad_ids
            sd[VOCAB_N_SAE_KEY] = bad_n_sae
            with pytest.raises(FeatureVocabError):
                feature_vocab_from_state_dict(sd)

    def test_a_1_ELEMENT_1D_n_sae_buffer_is_ACCEPTED(self):
        """The production encoder writes a 0-dim scalar, but a shape-(1,) buffer holds the
        SAME unambiguous number — ``.item()`` reads it identically. Rejecting it would be a
        gratuitous false positive, so the check is on the ELEMENT COUNT, not the rank."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        sd = _compact_state_dict(v)
        sd[VOCAB_N_SAE_KEY] = t.tensor([N], dtype=t.int64)
        n_sae, got = feature_vocab_from_state_dict(sd)
        assert n_sae == N
        assert got == v

    def test_a_1_ELEMENT_ids_buffer_is_ACCEPTED_the_V_equals_one_boundary(self):
        """V == 1 is the most extreme legal compaction. Its ids buffer is shape (1,) — it
        must not be mistaken for a corrupt scalar."""
        v = FeatureVocab(n_sae_features=N, original_ids=(9,))
        n_sae, got = feature_vocab_from_state_dict(_compact_state_dict(v))
        assert n_sae == N
        assert got == v and got.size == 1


# ===========================================================================
# assert_checkpoint_vocab_compatible — the 4-case table (spec §4.9.3)
# ===========================================================================


class TestCheckpointCompatibility:
    def test_legacy_model_and_legacy_checkpoint_is_fine(self):
        assert_checkpoint_vocab_compatible(None, _legacy_state_dict(), checkpoint_dir=CKPT)

    def test_compacted_model_and_matching_compacted_checkpoint_is_fine(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        assert_checkpoint_vocab_compatible(v, _compact_state_dict(v), checkpoint_dir=CKPT)

    def test_LEGACY_MASKED_checkpoint_into_compacted_model_raises_actionably(self):
        """The headline breakage (spec §6.5).

        PyTorch would emit a bare 'size mismatch'. Useless. The error must tell the operator
        something they can actually DO.

        This assertion used to be ``assert "feature_mask_freq_min" in msg``. It PASSED — and
        the advice it was pinning DID NOT WORK. See
        ``TestTheLegacyResumeRemedyActuallyWorks``: nulling the mask to get an uncompacted
        encoder also drops the ``bl_``/``wl_`` leaf from the composed ``checkpoint_dir``, so
        the reconfigured run addresses a DIFFERENT directory and never finds the checkpoint
        it set out to resume. Naming a knob is not the same as naming a remedy.
        """
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            assert_checkpoint_vocab_compatible(v, _legacy_state_dict(), checkpoint_dir=CKPT)
        msg = str(exc.value)
        low = msg.lower()
        assert CKPT in msg, "must name the checkpoint"
        assert "predates" in low or "uncompacted" in low, "must name WHY it cannot load"
        assert re.search(r"start fresh|re-?train", low), (
            "must name the one remedy that always works"
        )

    def test_compacted_checkpoint_into_unmasked_model_raises(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        with pytest.raises(FeatureVocabError, match="(?i)compacted"):
            assert_checkpoint_vocab_compatible(None, _compact_state_dict(v), checkpoint_dir=CKPT)

    def test_compacted_checkpoint_into_unmasked_model_names_the_recovery_cli(self):
        """The operator cannot proceed without the original mask, so the error must
        tell them how to recover it from the checkpoint itself (spec §4.9.4)."""
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            assert_checkpoint_vocab_compatible(None, _compact_state_dict(v), checkpoint_dir=CKPT)
        assert "dump_feature_mask" in str(exc.value)

    def test_different_ids_same_size_raises(self):
        """Same V ⇒ shapes match ⇒ load_state_dict would SUCCEED. Only this guard
        stands between you and a silently wrong dictionary."""
        model_v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        ckpt_v = FeatureVocab(n_sae_features=N, original_ids=(1, 4, 8, 12))
        assert model_v.size == ckpt_v.size  # shapes are identical!
        with pytest.raises(FeatureVocabError, match="differ|mismatch"):
            assert_checkpoint_vocab_compatible(
                model_v, _compact_state_dict(ckpt_v), checkpoint_dir=CKPT
            )

    def test_same_ids_different_n_sae_is_the_silent_wrong_model_bug(self):
        """Spec §4.9.3 — the bug my own first draft had.

        {whitelist, ids:[1,2,3]} on a 16k SAE and on a 65k SAE produce:
          - the same wl_<hash> checkpoint leaf (the hash covers mode+ids, NOT n_sae),
          - the same original_ids,
          - the same V=3 table shape.
        Ids match. Shapes match. Everything 'passes'. Only n_sae_features differs.
        Comparing ids alone is NOT sufficient.
        """
        model_v = FeatureVocab(n_sae_features=32, original_ids=(1, 2, 3))
        ckpt_v = FeatureVocab(n_sae_features=16, original_ids=(1, 2, 3))
        assert model_v.original_ids == ckpt_v.original_ids
        assert model_v.size == ckpt_v.size
        with pytest.raises(FeatureVocabError, match="n_sae_features"):
            assert_checkpoint_vocab_compatible(
                model_v, _compact_state_dict(ckpt_v), checkpoint_dir=CKPT
            )


# ===========================================================================
# FINDING 7 — the legacy-resume remedy has to actually WORK
# ===========================================================================


class TestTheLegacyResumeRemedyActuallyWorks:
    """The legacy-checkpoint error told the operator to resume a pre-#695 (uncompacted)
    checkpoint by nulling the ``feature_mask`` — so that no mask resolves and the encoder is
    built full-width.

    The *premise* is sound: a pre-#695 checkpoint is PHYSICALLY uncompacted, so resuming it
    genuinely does require that no mask resolve. The *instruction* is not, because the
    ``checkpoint_dir`` it would resume from is composed THROUGH ``append_mask_leaf``:

        checkpoint_dir = append_gradient_leaf(
            append_mask_leaf(.../d{depth}/s{seg}/{selection}, feature_mask), selection
        )                                     ^^^^^^^^^^^^

    and ``append_mask_leaf`` returns the path UNCHANGED when the mask is ``None``. Null the
    mask and the ``bl_<16hex>`` leaf simply vanishes from the path — so the reconfigured run
    addresses a DIFFERENT directory, finds no checkpoint there, and the operator has been
    sent on an errand that cannot succeed.

    A masked checkpoint LIVES under its mask leaf. There is no config knob that keeps the
    leaf while dropping compaction: the one input decides both.
    """

    def _msg(self) -> str:
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        with pytest.raises(FeatureVocabError) as exc:
            assert_checkpoint_vocab_compatible(v, _legacy_state_dict(), checkpoint_dir=CKPT)
        return str(exc.value)

    def test_PREMISE_nulling_the_mask_really_does_MOVE_the_checkpoint_dir(self):
        """Not a regression test — the FACT the message has to respect.

        Passes today and always did: it characterises ``append_mask_leaf``, which is
        correct. It is here because it is the premise the old remedy violated, and because
        a future 'simplification' of ``append_mask_leaf`` would silently make the rewritten
        message wrong again.
        """
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(1, 2))
        base = "/out/conditional_encdec_checkpoints/d12/s128/t2048.9"

        masked = append_mask_leaf(base, mask)
        unmasked = append_mask_leaf(base, None)

        assert masked == f"{base}/{mask.dirname()}", "a masked run lives UNDER the leaf"
        assert unmasked == base, "mask=None returns the path UNCHANGED — the leaf is gone"
        assert masked != unmasked, (
            "so 'null the feature_mask' silently RELOCATES the checkpoint_dir: the run would "
            "look for its checkpoint in a directory that has never held one"
        )

    def test_the_remedy_does_not_send_the_operator_to_a_DIFFERENT_directory(self):
        """If the message tells the operator to drop the mask, it MUST say that doing so
        also moves ``checkpoint_dir`` off the mask leaf. Otherwise the instruction reads as
        'do this and your resume works', and it does not."""
        msg = self._msg()
        low = msg.lower()

        mentions_dropping_the_mask = re.search(
            r"feature_mask(?:_freq_(?:min|max))?\s*[:=]?\s*null|null(?:ing)?\s+(?:the\s+)?"
            r"(?:feature_)?mask|no mask",
            low,
        )
        if mentions_dropping_the_mask:
            assert re.search(
                r"leaf|different director|another director|checkpoint_dir|"
                r"no longer|relocat|moves|move the",
                low,
            ), (
                "the message tells the operator to null the mask/band, but says nothing "
                "about the bl_/wl_ leaf disappearing from checkpoint_dir. append_mask_leaf "
                "returns the path unchanged for mask=None, so the reconfigured run looks in "
                "a DIFFERENT directory and cannot find this checkpoint at all. An "
                "instruction that cannot work is not a remedy."
            )

    def test_it_offers_a_remedy_that_ACTUALLY_works(self):
        """Passes today (the message already says 'start fresh'). Kept as a floor: the
        rewrite must not drop the one instruction that always succeeds."""
        assert re.search(r"start fresh|re-?train", self._msg().lower())

    def test_it_still_explains_the_SHAPE_conflict(self):
        msg = self._msg()
        assert "4" in msg and str(N) in msg, "must quote V and n_sae"
        assert "compact" in msg.lower()


# ===========================================================================
# assert_resume_feature_vocab_matches — the meta-side guard
# ===========================================================================


class TestResumeGuard:
    def test_no_configured_vocab_means_nothing_to_check(self):
        """Mirrors assert_resume_feature_mask_matches' asymmetry."""
        assert_resume_feature_vocab_matches(None, {"step": 5}, checkpoint_dir=CKPT)

    def test_matching_meta_passes(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        assert_resume_feature_vocab_matches(
            v, {"feature_vocab": v.metadata_dict()}, checkpoint_dir=CKPT
        )

    def test_a_MISSING_meta_key_is_FINE_the_buffers_are_the_record(self):
        """ABSENCE is fine; only DISAGREEMENT is a corruption signal.

        The persistent ``feature_vocab_ids`` / ``feature_vocab_n_sae`` buffers ARE the record —
        a compacted checkpoint describes itself — and by the time this guard runs, the resume
        path has already validated them against the configured vocab
        (``assert_checkpoint_vocab_compatible`` → ``load_state_dict(strict=True)`` →
        ``assert_vocab_consistent()``). So an absent mirror leaves nothing unverified, and
        raising on it merely discards a provably-recoverable checkpoint. Metas routinely get
        thinned, which is why the ancestor-meta machinery exists at all.

        This guard used to raise here, making the sidecar MANDATORY on every resume — and
        contradicting the contract that ``checkpoint_manager`` and ``dump_feature_mask`` both
        state in so many words.
        """
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        assert_resume_feature_vocab_matches(v, {"step": 5}, checkpoint_dir=CKPT)

    def test_a_None_meta_is_FINE_for_the_same_reason(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        assert_resume_feature_vocab_matches(v, None, checkpoint_dir=CKPT)

    def test_but_a_DISAGREEING_mirror_still_raises(self):
        """Permissive on absence must not mean permissive on contradiction: two halves of a
        checkpoint describing different models is exactly the corruption this exists to catch.
        """
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        other = FeatureVocab(n_sae_features=N, original_ids=(1, 4, 8, 12))
        with pytest.raises(FeatureVocabError, match="mismatch"):
            assert_resume_feature_vocab_matches(
                v, {"feature_vocab": other.metadata_dict()}, checkpoint_dir=CKPT
            )

    def test_fingerprint_mismatch_raises_and_names_the_dir(self):
        v = FeatureVocab(n_sae_features=N, original_ids=(0, 3, 7, 11))
        other = FeatureVocab(n_sae_features=N, original_ids=(1, 4, 8, 12))
        with pytest.raises(FeatureVocabError) as exc:
            assert_resume_feature_vocab_matches(
                v, {"feature_vocab": other.metadata_dict()}, checkpoint_dir=CKPT
            )
        assert CKPT in str(exc.value)

    def test_same_ids_different_n_sae_raises_here_too(self):
        """The n_sae guard must hold on the META path as well as the state-dict path."""
        model_v = FeatureVocab(n_sae_features=32, original_ids=(1, 2, 3))
        ckpt_v = FeatureVocab(n_sae_features=16, original_ids=(1, 2, 3))
        with pytest.raises(FeatureVocabError):
            assert_resume_feature_vocab_matches(
                model_v, {"feature_vocab": ckpt_v.metadata_dict()}, checkpoint_dir=CKPT
            )
