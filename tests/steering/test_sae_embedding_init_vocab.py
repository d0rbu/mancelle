"""SAE-backed encoder init under a compacted feature dictionary (#695 PR B, req 6).

``initialize_encoder_feature_embeddings_from_matrix`` currently derives
``n_sae_features`` from ``feature_weight.shape[0]`` and overwrites the table row-for-row
in raw-fid order. Once compacted, ``shape[0]`` is **V**, so:

* orienting ``W_dec`` against it would fail (W_dec has ``n_sae`` rows, not V), and
* row *j* of the compact table must receive the SAE direction of ORIGINAL id
  ``vocab.original_ids[j]`` — not of id *j*.

Get that wrong and every feature silently gets another feature's SAE direction.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch as t

from errors import FeatureVocabError
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder import sae_embedding_init
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.encoder import (
    _average_row_norm as _encoder_average_row_norm,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingInit,
    average_row_norm,
    initialize_encoder_feature_embeddings_from_matrix,
    orient_sae_decoder_matrix,
)
from steering.conditional.encoder_decoder.sae_embedding_init import (
    project_sae_decoder_matrix as _project_sae_decoder_matrix,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt

N_SAE, SAE_DIM, ENC_DIM = 64, 48, 32
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)

# svd needs V >= ENC_DIM. Its eigenbasis is drawn from a covariance of rank
# min(V, SAE_DIM), so a vocab narrower than the encoder dim makes ENC_DIM - V of the
# selected eigenvectors null-space directions and emits that many identically-zero
# columns. VOCAB (V=12 < ENC_DIM=32) is exactly such a vocab — every svd test in this
# file used it, and every one of them was silently producing a degenerate table. They
# now use a vocab wide enough to have a real basis; the narrow case is tested by
# TestSvdRefusesARankDeficientVocab, where it belongs.
SVD_ALLOWED = tuple(i for i in range(N_SAE) if i % 8 != 7)  # V = 56 >= ENC_DIM
SVD_VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=SVD_ALLOWED)

# The shapes that must ALL be bitwise. Chosen to hit every boundary of `select_rows`:
# V=1 at either end of the id space, SAE id 0 in and out of the vocabulary (it is the
# PADDING row, so it is the one row the reference and the compacted model can disagree
# about), and V=n_sae-1 — the near-identity, which is the widest vocab a model can
# actually be built with.
#
# The IDENTITY vocab (V == n_sae) used to be a param here. It is gone because
# ``EncoderDecoderGPT`` now REFUSES one at construction: a non-None FeatureVocab must always
# mean genuinely compacted, and an identity would register vocab buffers that no load path
# accepts. The gather property it was checking is still pinned, at the level where an
# identity remains legal — see ``TestTheIdentityGatherLivesOnFeatureVocabNow``.
BITWISE_SHAPES = [
    pytest.param((0,), id="V1-id0-the-padding-row"),
    pytest.param((63,), id="V1-last-id"),
    pytest.param((3, 7, 10), id="V3-no-id0"),
    pytest.param(ALLOWED, id="V12-with-id0"),
    pytest.param(tuple(range(1, N_SAE)), id="V63-everything-but-id0"),
]


def project_sae_decoder_matrix(
    matrix: t.Tensor,
    encoder_dim: int,
    mode: EncoderEmbeddingInit,
    *,
    seed: int,
    rows_are_feature_vocab: bool = False,
) -> t.Tensor:
    """Pass raw test values through the production beartype boundary."""
    return _project_sae_decoder_matrix(
        matrix,
        cast(PositiveInt, encoder_dim),
        mode,
        seed=seed,
        rows_are_feature_vocab=rows_are_feature_vocab,
    )


def _fresh(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    """A model on `meta`, materialised but NOT initialised."""
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                      n_kv_head=2, n_embd=64, window_pattern="L"),
            EncoderConfig(n_embd=ENC_DIM, depth=2, n_head=None, n_kv_head=None, head_dim=16),
            CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE, num_buckets=4, feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    return m


def _model(vocab: FeatureVocab | None) -> EncoderDecoderGPT:
    m = _fresh(vocab)
    t.manual_seed(0)
    m.init_weights()
    return m


def _apply_compute_dtype_cast(m: EncoderDecoderGPT, dtype: t.dtype) -> None:
    """The tail of ``FeatureEncoder.init_weights`` (encoder.py:466-469), verbatim.

    In production it is gated on ``device.type == "cuda"``, so on CPU it never fires and
    every existing byte-exactness test runs in fp32 — while the queued run is bf16 on an
    SM80+ GPU. Applying it explicitly is what lets a CPU test exercise the dtype the real
    run actually uses. The device gate is the only thing bypassed; the numerics are not.
    """
    if dtype == t.float32:
        return
    m.encoder.feature_embed.to(dtype=dtype)
    m.encoder.magnitude_embed.to(dtype=dtype)
    for ve in m.encoder.value_embeds.values():
        ve.to(dtype=dtype)


def _production_model(vocab: FeatureVocab | None, *, dtype: t.dtype, seed: int = 0):
    """A model built EXACTLY the way ``conditional_encdec_train.py`` builds one.

    Load-bearing detail: the train script calls ``init_weights()`` TWICE — once after
    ``to_empty`` (:1436) and again after re-seeding (:2449) — and only then runs SAE init
    (:2450). On CUDA the FIRST call casts the embeddings to bf16, so the SECOND runs the
    slice-init *in bf16*. Reproducing one call instead of two silently changes the dtype
    the rescale target is computed in. See ``TestTheStashMustDescribeTheLiveTable``.
    """
    m = _fresh(vocab)
    for _ in range(2):
        t.manual_seed(seed)
        m.init_weights()
        _apply_compute_dtype_cast(m, dtype)
    return m


def _w_dec(seed: int = 5) -> t.Tensor:
    """A W_dec whose row i is trivially identifiable: row i == i * ones + noise."""
    g = t.Generator().manual_seed(seed)
    base = t.arange(N_SAE, dtype=t.float32).unsqueeze(1).expand(N_SAE, SAE_DIM).contiguous()
    return base + 0.01 * t.randn(N_SAE, SAE_DIM, generator=g)


class TestCompactSaeInit:
    def test_random_mode_writes_a_V_row_table(self):
        m = _model(VOCAB)
        res = initialize_encoder_feature_embeddings_from_matrix(
            m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert m.encoder.feature_embed.weight.shape == (len(ALLOWED), ENC_DIM)
        assert res.n_sae_features == N_SAE, "must report the ORIGINAL width"
        assert res.vocab_size == len(ALLOWED)

    def test_compact_row_j_gets_the_direction_of_ORIGINAL_id_j(self):
        """The silent-corruption case: row j must carry original_ids[j]'s direction.

        RANDOM mode projects through a seeded, V-independent matrix, so the projected
        compact rows ARE the projected full rows for the same original ids (before the
        rescale). We can therefore check the mapping directly.
        """
        w = _w_dec()
        full_proj = project_sae_decoder_matrix(w, ENC_DIM, EncoderEmbeddingInit.RANDOM, seed=7)
        compact_proj = project_sae_decoder_matrix(
            VOCAB.select_rows(w), ENC_DIM, EncoderEmbeddingInit.RANDOM, seed=7
        )
        for j, orig in enumerate(ALLOWED):
            assert t.allclose(compact_proj[j], full_proj[orig], atol=1e-5), (
                f"compact row {j} must carry original id {orig}'s SAE direction"
            )

    def test_rescale_target_is_the_FULL_tables_norm_NOT_the_compact_ones(self):
        """The target is the number the UNCOMPACTED reference would have used.

        A V-row subset's mean row-norm is only a sample estimate of the full table's
        (measured: 2.1e-04 relative at V=35,275 of 65,536 on the real shape). Rescaling
        to the compact table's own norm would make compaction silently perturb the init
        — i.e. compaction would stop being purely representational. It must not.
        """
        ref, comp = _model(None), _model(VOCAB)
        full_norm = average_row_norm(ref.encoder.feature_embed.weight).item()
        compact_norm = average_row_norm(comp.encoder.feature_embed.weight).item()
        assert full_norm != compact_norm, (
            "the fixture is toothless unless the two norms genuinely differ"
        )
        res = initialize_encoder_feature_embeddings_from_matrix(
            comp, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert res.target_avg_norm == pytest.approx(full_norm, rel=1e-9)
        assert res.target_avg_norm != pytest.approx(compact_norm, rel=1e-9)

    def test_random_on_a_compacted_model_is_the_EXACT_row_subset_of_the_reference(self):
        """THE guarantee: compaction is purely representational under ``random``.

        Bitwise, not allclose. Projecting the FULL matrix and subsetting LAST means the
        compacted and reference runs execute the *identical* GEMM, so this cannot be
        broken by BLAS reducing differently at M=V vs M=n_sae.
        """
        ref, comp = _model(None), _model(VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(ALLOWED)]
        got = comp.encoder.feature_embed.weight
        assert t.equal(got, expected), (
            "compacted random-init must be BITWISE the reference's rows, not merely close"
        )

    def test_a_compacted_model_that_never_ran_init_weights_FAILS_LOUD(self):
        """The full-table norm is captured by ``init_weights()``. Without it we cannot
        reproduce the reference's rescale target — and we must never silently fall back
        to the compact table's norm, because that is exactly the bug this fix removes."""
        with t.device("meta"):
            m = EncoderDecoderGPT(
                GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                          n_kv_head=2, n_embd=64, window_pattern="L"),
                EncoderConfig(n_embd=ENC_DIM, depth=2, n_head=None, n_kv_head=None,
                              head_dim=16),
                CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
                n_sae_features=N_SAE, num_buckets=4, feature_vocab=VOCAB,
            )
        m.to_empty(device="cpu")  # deliberately NO init_weights()
        with pytest.raises(FeatureVocabError, match="(?i)init_weights"):
            initialize_encoder_feature_embeddings_from_matrix(
                m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
            )

    def test_svd_mode_works_on_a_compacted_table(self):
        m = _model(SVD_VOCAB)
        initialize_encoder_feature_embeddings_from_matrix(
            m, _w_dec(), EncoderEmbeddingInit.SVD, seed=7
        )
        assert m.encoder.feature_embed.weight.shape == (len(SVD_ALLOWED), ENC_DIM)
        assert t.isfinite(m.encoder.feature_embed.weight).all()

    def test_svd_covariance_is_over_the_SELECTED_rows_only(self):
        """Owner decision (requirement 6, literal): the principal directions are those
        of the SUB-dictionary, so a compacted SVD init deliberately DIFFERS from the
        row-subset of a full SVD init. Pinned so the choice cannot drift silently."""
        w = _w_dec()
        full = project_sae_decoder_matrix(w, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7)
        compact = project_sae_decoder_matrix(
            SVD_VOCAB.select_rows(w), ENC_DIM, EncoderEmbeddingInit.SVD, seed=7
        )
        subset_of_full = full[t.tensor(SVD_ALLOWED)]
        assert not t.allclose(compact, subset_of_full, atol=1e-3), (
            "SVD over the sub-dictionary should NOT equal the subset of the full SVD"
        )


class TestNormHelperAgreesWithSaeInit:
    """``encoder._average_row_norm`` duplicates ``sae_embedding_init.average_row_norm``
    (to keep ``sae_lens`` out of the encoder's import graph). A duplicate that DRIFTS
    would silently reintroduce the very rescale mismatch this fix removes, so pin them
    bit-for-bit."""

    @pytest.mark.parametrize("shape", [(64, 32), (7, 3), (1, 5), (129, 17)])
    def test_bitwise_identical(self, shape):
        g = t.Generator().manual_seed(11)
        m = t.randn(*shape, generator=g)
        assert t.equal(_encoder_average_row_norm(m), average_row_norm(m))

    def test_identical_in_bf16_too(self):
        g = t.Generator().manual_seed(12)
        m = t.randn(64, 32, generator=g).bfloat16()
        assert t.equal(_encoder_average_row_norm(m), average_row_norm(m))

    def test_the_encoder_captured_EXACTLY_the_full_tables_norm(self):
        """The stash must equal what the uncompacted reference's table actually has."""
        ref, comp = _model(None), _model(VOCAB)
        assert t.equal(
            comp.encoder.feature_embed_full_avg_row_norm,
            average_row_norm(ref.encoder.feature_embed.weight),
        )


class TestSvdTargetIsAlsoTheFullNorm:
    def test_svd_uses_the_full_tables_norm_as_its_target(self):
        """One target policy for both modes. Only the PROJECTION is sub-dictionary for
        svd — the rescale target is still the reference's, so the two modes cannot drift
        apart on the one axis that has a single right answer."""
        ref, comp = _model(None), _model(SVD_VOCAB)
        full_norm = average_row_norm(ref.encoder.feature_embed.weight).item()
        res = initialize_encoder_feature_embeddings_from_matrix(
            comp, _w_dec(), EncoderEmbeddingInit.SVD, seed=7
        )
        assert res.target_avg_norm == pytest.approx(full_norm, rel=1e-9)

    def test_but_svd_is_still_NOT_a_row_subset_of_the_reference(self):
        """Byte-exactness is out of reach for svd BY DESIGN (the eigenbasis depends on
        which rows feed the covariance). Pinned so nobody 'fixes' it into a full-basis
        SVD and quietly reverses the owner's decision."""
        ref, comp = _model(None), _model(SVD_VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(ref, w, EncoderEmbeddingInit.SVD, seed=7)
        initialize_encoder_feature_embeddings_from_matrix(comp, w, EncoderEmbeddingInit.SVD, seed=7)
        subset = ref.encoder.feature_embed.weight[t.tensor(SVD_ALLOWED)]
        assert not t.allclose(comp.encoder.feature_embed.weight, subset, atol=1e-3)


class TestUncompactedIsUnchanged:
    def test_no_vocab_path_still_writes_the_full_table(self):
        m = _model(None)
        res = initialize_encoder_feature_embeddings_from_matrix(
            m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert m.encoder.feature_embed.weight.shape == (N_SAE, ENC_DIM)
        assert res.n_sae_features == N_SAE
        assert res.vocab_size == N_SAE

    def test_no_vocab_result_is_bitwise_reproducible(self):
        a, b = _model(None), _model(None)
        initialize_encoder_feature_embeddings_from_matrix(
            a, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            b, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert t.equal(a.encoder.feature_embed.weight, b.encoder.feature_embed.weight)


# ===========================================================================
# `random` is BITWISE — across every shape, every dtype, and a transposed W_dec
# ===========================================================================


class TestTheIdentityGatherLivesOnFeatureVocabNow:
    """The identity boundary, re-homed.

    ``TestRandomIsBitwiseAtEveryShape`` used to drive an IDENTITY ``FeatureVocab`` through
    ``EncoderDecoderGPT`` and assert the gather reproduced the whole reference table. Its
    own docstring justified this as "``from_mask`` normalises this to ``None``, but the
    class is constructible directly" — which is precisely the hole that got closed: the
    constructor now REFUSES an identity, because building one registers vocab buffers that
    ``feature_vocab_from_state_dict`` (rightly) rejects, i.e. it produced a model that could
    be saved and never loaded.

    ``FeatureVocab`` itself still permits an identity — it must, since ``from_mask`` builds
    one before normalising it away — so that is where the gather property belongs. Nothing
    is lost; it is asserted one layer down, on the function that actually does the gather.
    """

    IDENTITY = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(N_SAE)))

    def test_the_model_constructor_REFUSES_it(self):
        assert self.IDENTITY.is_identity
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            _model(self.IDENTITY)

    def test_select_rows_on_an_identity_is_the_WHOLE_table_bitwise(self):
        """The property the deleted model-level test was really about."""
        full = t.randn(N_SAE, ENC_DIM, generator=t.Generator().manual_seed(7))
        assert t.equal(self.IDENTITY.select_rows(full), full)

    def test_the_NEAR_identity_still_goes_through_the_MODEL(self):
        """V = n_sae - 1 is constructible, and it is the widest shape a model can hold — so
        the widest end of the gather is still covered end-to-end (see BITWISE_SHAPES)."""
        near = FeatureVocab(
            n_sae_features=N_SAE, original_ids=tuple(range(1, N_SAE))
        )
        assert not near.is_identity
        assert _model(near).encoder.feature_embed.weight.shape[0] == N_SAE - 1


class TestRandomIsBitwiseAtEveryShape:
    """One V is not a proof. ``select_rows`` is a gather, and gathers go wrong at the
    ends: V=1, the identity vocab, and — the one that actually matters — whether SAE id 0
    (the PADDING id) is inside the vocabulary. Row 0 of a compacted table is
    ``feature_embed[original_ids[0]]``, which coincides with the reference's row 0 only
    when 0 survives the mask, so both cases must be pinned.
    """

    @pytest.mark.parametrize("ids", BITWISE_SHAPES)
    def test_random_is_the_EXACT_row_subset_of_the_reference(self, ids):
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=ids)
        ref, comp = _model(None), _model(vocab)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(ids)]
        got = comp.encoder.feature_embed.weight
        assert got.shape == (len(ids), ENC_DIM)
        assert t.equal(got, expected), (
            f"V={len(ids)}: compacted random-init must be BITWISE the reference's rows"
        )

    def test_a_row_is_NOT_bitwise_equal_to_a_DIFFERENT_features_row(self):
        """The negative control. Without it, a `select_rows` that returned (say) the first
        V rows would still pass every test above IF the reference's rows happened to be
        near-identical — so prove the rows are actually distinguishable."""
        ref = _model(None)
        initialize_encoder_feature_embeddings_from_matrix(
            ref, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        w = ref.encoder.feature_embed.weight
        assert not t.equal(w[ALLOWED[0]], w[ALLOWED[1]])
        assert not t.equal(w[0], w[1])


class TestRandomIsBitwiseInBf16Too:
    """Every other byte-exactness test in this repo runs fp32 on CPU.

    The queued run is **bf16 on an SM80+ GPU**, where ``FeatureEncoder.init_weights``
    casts ``feature_embed`` to ``COMPUTE_DTYPE`` (encoder.py:466). Under that cast the
    rescale target is computed from a bf16 table for the reference and from the
    ``init_weights`` stash for the compacted model — and those are the SAME number only if
    the stash was itself captured in bf16. Nothing in an fp32 test can see that.
    """

    @pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
    def test_random_is_the_EXACT_row_subset_under_the_production_build(self, dtype):
        ref = _production_model(None, dtype=dtype)
        comp = _production_model(VOCAB, dtype=dtype)
        assert ref.encoder.feature_embed.weight.dtype == dtype
        assert comp.encoder.feature_embed.weight.dtype == dtype

        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(ALLOWED)]
        assert t.equal(comp.encoder.feature_embed.weight, expected)

    @pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
    def test_the_two_models_agree_on_the_rescale_TARGET_bit_for_bit(self, dtype):
        """The single number the two runs can disagree about. Isolated from the GEMM so a
        failure names the cause instead of just 'the rows differ'."""
        ref = _production_model(None, dtype=dtype)
        comp = _production_model(VOCAB, dtype=dtype)
        ref_target = average_row_norm(ref.encoder.feature_embed.weight)
        comp_target = comp.encoder.feature_embed_full_avg_row_norm
        assert t.equal(ref_target, comp_target)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_on_REAL_cuda_with_the_REAL_dtype_cast(self):
        """No simulation: let ``init_weights`` fire its own device-gated cast."""
        def build(vocab):
            with t.device("meta"):
                m = EncoderDecoderGPT(
                    GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                              n_kv_head=2, n_embd=64, window_pattern="L"),
                    EncoderConfig(n_embd=ENC_DIM, depth=2, n_head=None, n_kv_head=None,
                                  head_dim=16),
                    CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
                    n_sae_features=N_SAE, num_buckets=4, feature_vocab=vocab,
                )
            m.to_empty(device="cuda")
            for _ in range(2):  # the train script's double init
                t.manual_seed(0)
                t.cuda.manual_seed(0)
                m.init_weights()
            return m

        ref, comp = build(None), build(VOCAB)
        assert ref.encoder.feature_embed.weight.dtype == COMPUTE_DTYPE
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(ALLOWED, device="cuda")]
        assert t.equal(comp.encoder.feature_embed.weight, expected)


class TestTheStashMustDescribeTheLiveTable:
    """``feature_embed_full_avg_row_norm`` is documented as "the FULL (n_sae, d) feature
    table's average row norm". The compacted model rescales to it; the reference rescales
    to ``average_row_norm(its own live table)``. Byte-exactness is exactly the claim that
    those two are the same number — so the stash must describe the table AS IT STANDS when
    ``init_weights()`` returns, not some earlier version of it.

    This is the sentinel for the dtype coupling. ``init_weights`` captures the stash
    (encoder.py:434) and only THEN casts the table to bf16 (encoder.py:466), so a single
    call leaves a stash describing a table that no longer exists. Production survives
    only because ``conditional_encdec_train.py`` calls ``init_weights()`` twice, and the
    second call re-derives the stash from the already-cast table. Delete that redundant
    first call — an obvious cleanup — and ``random`` silently stops being bitwise on GPU.
    """

    @pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
    def test_the_stash_equals_the_live_tables_norm_on_an_uncompacted_model(self, dtype):
        ref = _production_model(None, dtype=dtype)
        assert t.equal(
            ref.encoder.feature_embed_full_avg_row_norm,
            average_row_norm(ref.encoder.feature_embed.weight),
        )

    @pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
    def test_a_compacted_models_stash_equals_the_REFERENCES_live_table_norm(self, dtype):
        ref = _production_model(None, dtype=dtype)
        comp = _production_model(VOCAB, dtype=dtype)
        assert t.equal(
            comp.encoder.feature_embed_full_avg_row_norm,
            average_row_norm(ref.encoder.feature_embed.weight),
        )

    @pytest.mark.skipif(
        not t.cuda.is_available() or COMPUTE_DTYPE == t.float32,
        reason="the dtype cast in FeatureEncoder.init_weights is CUDA + non-fp32 only",
    )
    def test_ONE_init_weights_leaves_a_consistent_stash(self):
        """A SINGLE ``init_weights()`` must leave a stash describing the LIVE table.

        This was a real latent bug, and it only bit on the hardware the queued run uses.
        The stash was captured BEFORE ``init_weights`` casts ``feature_embed`` to
        COMPUTE_DTYPE (a CUDA-only cast), so one call left it describing a discarded fp32
        table while the uncompacted reference's SAE-init read its live bf16 one. Measured
        on an RTX 4090: rescale targets 5.576264 vs 5.576411, and 5 of 384 elements of the
        resulting compacted table then differed from the reference's rows.

        Production survived only by ACCIDENT — ``conditional_encdec_train.py`` calls
        ``init_weights()`` twice (:1436 and :2449) and the second call re-derived the stash
        in bf16. Deleting that redundant first call is an obvious cleanup that would have
        silently broken the headline bitwise guarantee on GPU, and no CPU test would have
        noticed. The capture now happens after the cast, so one call is enough and the
        coupling is gone.
        """
        with t.device("meta"):
            m = EncoderDecoderGPT(
                GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                          n_kv_head=2, n_embd=64, window_pattern="L"),
                EncoderConfig(n_embd=ENC_DIM, depth=2, n_head=None, n_kv_head=None,
                              head_dim=16),
                CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
                n_sae_features=N_SAE, num_buckets=4, feature_vocab=None,
            )
        m.to_empty(device="cuda")
        t.manual_seed(0)
        t.cuda.manual_seed(0)
        m.init_weights()  # ONCE — as any new caller would naturally write it

        assert t.equal(
            m.encoder.feature_embed_full_avg_row_norm,
            average_row_norm(m.encoder.feature_embed.weight),
        ), "the stash describes a table that no longer exists"


class TestTheProjectionRunsAtFULLWIDTH:
    """``random`` subsets LAST, on purpose: the compacted and reference runs then execute
    the IDENTICAL GEMM. Subsetting first is mathematically equal but a different-shaped
    one (M=V vs M=n_sae), which leaves byte-exactness hostage to how BLAS reduces.

    Today's BLAS happens to agree, so a behavioural test CANNOT distinguish the two
    orderings — a regression to subset-first would keep every bitwise test above green and
    only bite on some future kernel, machine, or thread count. This pins the ordering
    itself, which is the only thing that actually guarantees it.
    """

    def _projection_shapes(self, vocab, mode) -> list[tuple[int, ...]]:
        seen: list[tuple[int, ...]] = []
        real = sae_embedding_init.project_sae_decoder_matrix

        def spy(matrix, *args, **kwargs):
            seen.append(tuple(matrix.shape))
            return real(matrix, *args, **kwargs)

        with patch.object(sae_embedding_init, "project_sae_decoder_matrix", spy):
            initialize_encoder_feature_embeddings_from_matrix(
                _model(vocab), _w_dec(), mode, seed=7
            )
        return seen

    def test_random_projects_the_FULL_n_sae_by_sae_dim_matrix(self):
        assert self._projection_shapes(VOCAB, EncoderEmbeddingInit.RANDOM) == [
            (N_SAE, SAE_DIM)
        ]

    def test_and_the_uncompacted_reference_projects_the_very_same_shape(self):
        assert self._projection_shapes(None, EncoderEmbeddingInit.RANDOM) == [
            (N_SAE, SAE_DIM)
        ]

    def test_svd_is_the_deliberate_OPPOSITE_and_projects_only_the_SUBSET(self):
        """Requirement 6, literal: the eigenbasis must be the SUB-dictionary's. Pinned so
        the asymmetry between the two modes is a decision, not an accident."""
        assert self._projection_shapes(SVD_VOCAB, EncoderEmbeddingInit.SVD) == [
            (len(SVD_ALLOWED), SAE_DIM)
        ]


class TestATransposedWdecIsStillOrientedCorrectly:
    """SAELens stores ``W_dec`` as ``(n_sae, sae_dim)``, but ``orient_sae_decoder_matrix``
    accepts the transpose. If it ever oriented the wrong way, every feature would receive
    a linear combination of all features instead of its own direction — and the result
    would still be finite, still be the right shape, and still rescale cleanly."""

    def test_the_transpose_gives_a_BITWISE_identical_table(self):
        a, b = _model(VOCAB), _model(VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            a, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            b, w.T.contiguous(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert t.equal(a.encoder.feature_embed.weight, b.encoder.feature_embed.weight)

    def test_and_it_still_matches_the_uncompacted_references_rows(self):
        ref, comp = _model(None), _model(VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w.T.contiguous(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(ALLOWED)]
        assert t.equal(comp.encoder.feature_embed.weight, expected)

    def test_orient_is_not_a_no_op_on_the_transpose(self):
        """Guards the test above from passing vacuously: N_SAE != SAE_DIM, so the two
        orientations really are different tensors."""
        assert N_SAE != SAE_DIM, "otherwise orientation is ambiguous and untestable"
        w = _w_dec()
        assert t.equal(orient_sae_decoder_matrix(w, N_SAE), w)
        assert t.equal(orient_sae_decoder_matrix(w.T.contiguous(), N_SAE), w)

    def test_a_matrix_matching_NEITHER_axis_raises(self):
        with pytest.raises(ValueError, match="must include n_sae_features"):
            orient_sae_decoder_matrix(t.zeros(N_SAE + 1, SAE_DIM), N_SAE)


class TestTheRescaleTargetIsEXACT:
    def test_the_target_is_BITWISE_the_full_tables_norm_not_merely_close(self):
        """The existing test uses ``pytest.approx(rel=1e-9)``. The defect it guards against
        (rescaling to the V-row table's norm) is 2.1e-04 relative at the real shape — a
        thousand times larger than that tolerance, so ``approx`` would have caught it. But
        the CLAIM is bitwise, and only ``==`` tests the claim."""
        ref, comp = _model(None), _model(VOCAB)
        res = initialize_encoder_feature_embeddings_from_matrix(
            comp, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert res.target_avg_norm == average_row_norm(ref.encoder.feature_embed.weight).item()

    def test_the_COMPACT_table_is_NOT_rescaled_to_the_target(self):
        """The discriminator between 'project full, subset last' and 'subset, then project
        and rescale the subset'. Both produce a V-row table of the right shape with finite
        values; only the second forces the SUBSET's average row norm ONTO the target.

        ``random`` rescales the FULL matrix and subsets afterwards, so the compact table's
        own average row norm is a SAMPLE of the target and lands materially away from it.

        The assertion is on the MAGNITUDE of the gap, not on ``!=``. A bare ``!=`` does not
        discriminate: rescale-the-subset still misses the target by ~1e-07 relative (the
        scale-then-renorm round-trip is not exact in fp32), so ``!=`` is satisfied by pure
        floating-point noise and stays green under the very regression it claims to catch.
        Measured here: 2.3e-01 relative when subsetting LAST, 8.5e-08 when subsetting
        first — five orders of magnitude apart, so 1e-3 separates them with room to spare.
        """
        res_comp = initialize_encoder_feature_embeddings_from_matrix(
            _model(VOCAB), _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        gap = abs(res_comp.final_avg_norm - res_comp.target_avg_norm) / res_comp.target_avg_norm
        assert gap > 1e-3, (
            f"the compact table's own norm sits on the target (relative gap {gap:.2e}), "
            "i.e. the subset happened BEFORE the rescale — the non-bitwise ordering"
        )
        # ...while the uncompacted reference, whose table IS the rescaled matrix, lands on it.
        res_ref = initialize_encoder_feature_embeddings_from_matrix(
            _model(None), _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert res_ref.final_avg_norm == pytest.approx(res_ref.target_avg_norm, rel=1e-6)

    def test_the_reported_vocab_size_is_V_and_n_sae_features_is_the_ORIGINAL_width(self):
        res = initialize_encoder_feature_embeddings_from_matrix(
            _model(VOCAB), _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert (res.vocab_size, res.n_sae_features) == (len(ALLOWED), N_SAE)
        assert res.sae_dim == SAE_DIM
        assert res.encoder_dim == ENC_DIM


class TestFailLoudWhenTheNormWasNeverCaptured:
    @pytest.mark.parametrize(
        "mode", [EncoderEmbeddingInit.RANDOM, EncoderEmbeddingInit.SVD], ids=["random", "svd"]
    )
    def test_BOTH_modes_raise(self, mode):
        """``svd`` reads the same stash (the target policy is shared). The existing test
        only covers ``random``, so a fallback reintroduced on the svd branch would be
        invisible."""
        with pytest.raises(FeatureVocabError, match="never captured"):
            initialize_encoder_feature_embeddings_from_matrix(
                _fresh(SVD_VOCAB), _w_dec(), mode, seed=7
            )

    def test_the_message_names_init_weights_and_the_required_ORDER(self):
        with pytest.raises(FeatureVocabError) as exc:
            initialize_encoder_feature_embeddings_from_matrix(
                _fresh(VOCAB), _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
            )
        msg = str(exc.value)
        assert "init_weights() has not run" in msg
        assert "BEFORE initialize_encoder_feature_embeddings_from_sae" in msg

    def test_it_does_NOT_silently_fall_back_to_the_compact_tables_norm(self):
        """The bug this guard replaced. The compact table is right there and its norm is
        *plausible* — so prove that nothing was written, rather than merely that something
        was raised.

        The table is filled with a real draw rather than left as ``to_empty`` garbage, for
        two reasons. It makes the trap REAL: ``average_row_norm(feature_weight)`` is now
        finite, positive and entirely usable, so a fallback would succeed silently — which
        is exactly the failure mode under test. And it makes the assertion SOUND:
        ``t.equal`` is False whenever either tensor holds a NaN, so comparing uninitialized
        memory against its own clone fails at random depending on what was in the page.
        """
        m = _fresh(VOCAB)
        with t.no_grad():
            m.encoder.feature_embed.weight.normal_(mean=0.0, std=1.0)
        plausible = average_row_norm(m.encoder.feature_embed.weight)
        assert t.isfinite(plausible) and plausible > 0, (
            "the fixture is toothless unless the compact table's norm is a usable fallback"
        )

        before = m.encoder.feature_embed.weight.clone()
        with pytest.raises(FeatureVocabError):
            initialize_encoder_feature_embeddings_from_matrix(
                m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
            )
        assert t.equal(m.encoder.feature_embed.weight, before), "the table was overwritten"

    def test_the_UNCOMPACTED_path_never_READS_the_stash(self):
        """Invariant I5: zero added ops on the no-mask path. An uncompacted model reads its
        own live table, so poisoning the stash must not matter."""
        m = _model(None)
        m.encoder._feature_embed_full_avg_row_norm = None  # the property now raises if read
        res = initialize_encoder_feature_embeddings_from_matrix(
            m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert res.vocab_size == N_SAE


class TestSvdStaysNonBitwiseByDesign:
    def test_svd_compacted_is_NOT_the_row_subset_of_the_reference(self):
        """Already pinned elsewhere with ``allclose``; restated with ``t.equal`` so the
        claim 'svd is NOT bitwise' is tested as literally as the claim 'random IS'."""
        ref, comp = _model(None), _model(SVD_VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(ref, w, EncoderEmbeddingInit.SVD, seed=7)
        initialize_encoder_feature_embeddings_from_matrix(comp, w, EncoderEmbeddingInit.SVD, seed=7)
        subset = ref.encoder.feature_embed.weight[t.tensor(SVD_ALLOWED)]
        assert not t.equal(comp.encoder.feature_embed.weight, subset)

    def test_but_svd_IS_deterministic(self):
        """Non-bitwise vs the reference must not be mistaken for non-reproducible. Two
        compacted runs with the same seed must agree exactly, or resumes and reruns drift."""
        a, b = _model(SVD_VOCAB), _model(SVD_VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(a, w, EncoderEmbeddingInit.SVD, seed=7)
        initialize_encoder_feature_embeddings_from_matrix(b, w, EncoderEmbeddingInit.SVD, seed=7)
        assert t.equal(a.encoder.feature_embed.weight, b.encoder.feature_embed.weight)

    def test_svd_REFUSES_a_RANK_DEFICIENT_sub_dictionary(self):
        """This test used to assert the OPPOSITE, and its own docstring named the defect:

            "V=1: the covariance of a single row is rank-1, so ENC_DIM-1 of the
             eigenvectors it projects onto are arbitrary null-space directions. It must
             still produce a finite, non-degenerate row..."

        It then checked only ``isfinite()`` and ``norm() > 0`` — both of which are TRUE for
        a table with 31 of its 32 columns identically zero. It passed for the wrong reason,
        and it made a structurally degenerate init look like a supported configuration.

        A rank-deficient basis is not something to survive. It is something to refuse.
        """
        narrow = FeatureVocab(n_sae_features=N_SAE, original_ids=(7,))
        m = _model(narrow)
        with pytest.raises(ValueError, match="(?i)rank|null-space"):
            initialize_encoder_feature_embeddings_from_matrix(
                m, _w_dec(), EncoderEmbeddingInit.SVD, seed=7
            )

    def test_svd_is_bitwise_reproducible_from_a_transposed_W_dec_too(self):
        a, b = _model(SVD_VOCAB), _model(SVD_VOCAB)
        w = _w_dec()
        initialize_encoder_feature_embeddings_from_matrix(a, w, EncoderEmbeddingInit.SVD, seed=7)
        initialize_encoder_feature_embeddings_from_matrix(
            b, w.T.contiguous(), EncoderEmbeddingInit.SVD, seed=7
        )
        assert t.equal(a.encoder.feature_embed.weight, b.encoder.feature_embed.weight)


class TestSvdRefusesARankDeficientVocab:
    """The svd rank guard was written when the row count was ALWAYS ``n_sae``.

    ``project_sae_decoder_matrix`` guards only ``encoder_dim <= sae_dim``. That was
    sufficient while ``matrix`` had ``n_sae`` rows (16,384 / 65,536), because
    ``rank(matrix.T @ matrix)`` was then always ``sae_dim``. Since #695 the svd path is fed
    the **V-row subset**, so the rank is ``min(V, sae_dim)`` — and a whitelist with
    ``V < encoder_dim`` makes ``encoder_dim - V`` of the selected eigenvectors NULL-SPACE
    directions with ~zero eigenvalues.

    The result is a structurally degenerate table — whole columns identically ~0 — and
    nothing catches it: ``average_row_norm`` is finite, ``normalize_to_average_row_norm``'s
    ``current_avg_norm <= 0`` check does not trip, and the table is rescaled to the full
    table's norm and reported as a normal init.

    Measured before the guard (SAE_DIM=48, encoder_dim=32): V=16 -> 16 of 32 columns dead;
    V=4 -> 28 of 32 dead. Silently.
    """

    @pytest.mark.parametrize("v", [1, 4, 16, ENC_DIM - 1])
    def test_a_vocab_narrower_than_the_encoder_dim_RAISES(self, v):
        matrix = t.randn(v, SAE_DIM, generator=t.Generator().manual_seed(0))
        with pytest.raises(ValueError, match="(?i)rank|rows|whitelist"):
            project_sae_decoder_matrix(matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7)

    @pytest.mark.parametrize("v", [ENC_DIM, ENC_DIM + 1, SAE_DIM, N_SAE])
    def test_a_vocab_at_or_above_the_encoder_dim_is_FINE(self, v):
        """The boundary is exact: V == encoder_dim is full-rank and must be allowed."""
        matrix = t.randn(v, SAE_DIM, generator=t.Generator().manual_seed(0))
        out = project_sae_decoder_matrix(matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7)
        col_norms = out.norm(dim=0)
        assert int((col_norms < 1e-4 * col_norms.max()).sum()) == 0, (
            "no column may be degenerate at or above the rank boundary"
        )

    def test_the_message_names_V_the_encoder_dim_and_the_remedy(self):
        matrix = t.randn(4, SAE_DIM, generator=t.Generator().manual_seed(0))
        with pytest.raises(ValueError) as exc:
            project_sae_decoder_matrix(matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7)
        msg = str(exc.value)
        assert "4" in msg and str(ENC_DIM) in msg
        assert "random" in msg.lower(), "the operator must be told the escape hatch"

    def test_the_END_TO_END_svd_init_refuses_a_narrow_whitelist(self):
        """Through the real entry point, not just the projection helper."""
        narrow = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(4)))
        model = _model(narrow)
        with pytest.raises(ValueError, match="(?i)rank|rows|whitelist"):
            initialize_encoder_feature_embeddings_from_matrix(
                model, _w_dec(), EncoderEmbeddingInit.SVD, seed=7
            )

    def test_RANDOM_is_unaffected_by_a_narrow_vocab(self):
        """random projects through a seeded, rank-independent matrix — no such hazard."""
        narrow = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(4)))
        model = _model(narrow)
        initialize_encoder_feature_embeddings_from_matrix(
            model, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        w = model.encoder.feature_embed.weight
        col_norms = w.norm(dim=0)
        assert int((col_norms < 1e-4 * col_norms.max()).sum()) == 0


class TestTheSaeInitLogIsReadableOnACompactedRun:
    """The three logged norms exist to answer one question — "is the rescale working?" —
    and on a compacted run they could not.

    * The log reported ``n_sae_features`` (the ORIGINAL width) and never ``vocab_size``, so
      a compacted cold start said ``n_sae_features=65536`` when only V rows were written.
    * ``final_avg_norm`` is measured over the V WRITTEN rows while ``target_avg_norm`` is
      deliberately the FULL table's norm. Pre-#695 they agreed by construction; now they do
      not, and nothing said why — so an operator could not tell a working rescale from a
      subtly broken init, which is precisely the diagnostic those numbers are for.

    The log is spied at ``logger.info`` rather than through a loguru sink: the message is a
    ``str.format`` TEMPLATE plus positional args, so rendering it here is both simpler and a
    stricter check (a template that forgets to consume an arg still renders "fine").
    """

    def _log_text(self, vocab: FeatureVocab | None) -> str:
        """Drive the REAL entry point (which is where the log lives), stubbing only the
        SAE load — the one thing that would otherwise pull a multi-GB dictionary off disk."""
        m = _model(vocab)
        captured: list[str] = []

        def spy(template, *args, **kwargs):
            captured.append(str(template).format(*args, **kwargs))

        cfg = {
            "initialize_encoder_embeddings_from_sae": "random",
            "sae": {"model": "stub", "width": "16k", "layer": 8},
        }
        with (
            patch.object(sae_embedding_init.logger, "info", spy),
            patch.object(
                sae_embedding_init, "load_sae_decoder_matrix_from_cfg", lambda _c: _w_dec()
            ),
        ):
            sae_embedding_init.maybe_initialize_encoder_feature_embeddings_from_sae(
                m, cfg, seed=7
            )
        return "\n".join(captured)

    def test_the_result_reports_the_rows_it_actually_WROTE(self):
        m = _model(VOCAB)
        res = initialize_encoder_feature_embeddings_from_matrix(
            m, _w_dec(), EncoderEmbeddingInit.RANDOM, seed=7
        )
        assert res.vocab_size == VOCAB.size == m.encoder.feature_embed.weight.shape[0]
        assert res.n_sae_features == N_SAE
        assert res.vocab_size != res.n_sae_features, "the fixture must be compacted"

    def test_the_LOG_LINE_states_the_written_row_count(self):
        text = self._log_text(VOCAB)
        assert f"vocab_size={VOCAB.size}" in text, (
            f"the log must state the {VOCAB.size} rows it wrote, not only "
            f"n_sae_features={N_SAE}: {text!r}"
        )
        assert f"n_sae_features={N_SAE}" in text, "the ORIGINAL width must stay visible"

    def test_the_log_SAYS_the_encoder_was_compacted(self):
        """Otherwise `final_avg_norm != target_avg_norm` reads as a bug when it is design."""
        assert "COMPACTED" in self._log_text(VOCAB)

    def test_an_UNCOMPACTED_run_says_so_and_does_not_cry_wolf(self):
        text = self._log_text(None)
        assert "COMPACTED" not in text
        assert f"n_sae_features={N_SAE}" in text


class TestTheRankGuardDiagnosesTheRIGHTModel:
    """#695 review FINDING 7. The guard fires for two different reasons; it said one.

    ``project_sae_decoder_matrix`` sees only a matrix. Whether its ``rows`` are a
    *vocabulary* (the mask's allowed universe, subset in ahead of the projection — the
    compacted svd path) or the *whole SAE dictionary* (every other caller) is knowledge
    only its caller has.

    The guard computed ``limiting = "vocabulary size" if n_rows < sae_dim else "SAE
    decoder dimension"`` and then unconditionally told the operator to **"widen the
    mask"**. On an UNCOMPACTED model ``rows`` is ``n_sae_features`` — the entire
    dictionary — so it sent an operator who has NO mask off to widen one. And on a
    compacted model whose ``sae_dim`` is the real bottleneck, widening the mask cannot
    help either: the message must say so instead of prescribing a no-op.

    It must still RAISE in every case. Only the diagnosis changes.
    """

    # V < ENC_DIM: the mask's allowed universe is what runs out of rank.
    def _compacted_msg(self) -> str:
        matrix = t.randn(4, SAE_DIM, generator=t.Generator().manual_seed(0))
        with pytest.raises(ValueError) as exc:
            project_sae_decoder_matrix(
                matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7, rows_are_feature_vocab=True
            )
        return str(exc.value)

    # rows == the whole dictionary, and it is narrower than ENC_DIM. No mask exists.
    def _uncompacted_msg(self) -> str:
        matrix = t.randn(4, SAE_DIM, generator=t.Generator().manual_seed(0))
        with pytest.raises(ValueError) as exc:
            project_sae_decoder_matrix(matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7)
        return str(exc.value)

    def test_the_COMPACTED_case_still_says_widen_the_mask(self):
        msg = self._compacted_msg()
        assert "feature_mask" in msg, "the mask IS the binding constraint here"
        assert "widen" in msg.lower()

    def test_the_UNCOMPACTED_case_must_NOT_say_widen_the_mask(self):
        """The bug. There is no mask on this model — 'widen the feature_mask' is nonsense
        and sends the operator chasing a config key they never set."""
        msg = self._uncompacted_msg()
        assert "widen" not in msg.lower(), (
            "told an operator with NO feature_mask to widen one. `rows` here is "
            "n_sae_features — the WHOLE SAE dictionary — not a mask's allowed universe."
        )

    def test_the_UNCOMPACTED_case_names_the_dictionary_and_a_REAL_remedy(self):
        msg = self._uncompacted_msg()
        low = msg.lower()
        assert "dictionary" in low, "must say the SAE dictionary itself is what ran out"
        assert "not" in low and "compact" in low, (
            "must state the model is NOT compacted — otherwise the operator cannot tell "
            "which of the two failures they are looking at"
        )
        assert "n_embd" in msg or "encoder_dim" in msg, (
            "the only real remedies are a smaller encoder dim or a wider SAE"
        )
        assert "random" in low, "the rank-independent escape hatch"

    def test_when_sae_dim_binds_the_message_does_not_prescribe_a_no_op(self):
        """V=64 >= ENC_DIM, but sae_dim=8 < ENC_DIM. Widening the mask changes nothing."""
        matrix = t.randn(N_SAE, 8, generator=t.Generator().manual_seed(0))
        with pytest.raises(ValueError) as exc:
            project_sae_decoder_matrix(
                matrix, ENC_DIM, EncoderEmbeddingInit.SVD, seed=7, rows_are_feature_vocab=True
            )
        msg = str(exc.value)
        assert "SAE decoder dimension is the binding constraint" in msg
        low = msg.lower()
        assert "cannot help" in low or "will not help" in low, (
            "the mask is NOT the constraint here — the message must say widening it is a "
            "no-op rather than sending the operator to do it"
        )

    def test_BOTH_cases_still_RAISE(self):
        """The diagnosis changed; the refusal did not."""
        for flag in (True, False):
            with pytest.raises(ValueError, match="(?i)rank|null-space"):
                project_sae_decoder_matrix(
                    t.randn(4, SAE_DIM),
                    ENC_DIM,
                    EncoderEmbeddingInit.SVD,
                    seed=7,
                    rows_are_feature_vocab=flag,
                )

    def test_the_production_path_flags_a_compacted_svd_run_as_a_vocabulary(self):
        """The wiring. The end-to-end svd path subsets to V BEFORE projecting, so it — and
        only it — must tell the guard that its rows are a vocabulary."""
        narrow = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(4)))
        with pytest.raises(ValueError) as exc:
            initialize_encoder_feature_embeddings_from_matrix(
                _model(narrow), _w_dec(), EncoderEmbeddingInit.SVD, seed=7
            )
        assert "widen" in str(exc.value).lower(), (
            "the real entry point subset to V rows but did not tell the guard, so the "
            "operator got the 'you have no mask' diagnosis for a masked model"
        )
