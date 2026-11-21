"""Encoder feature-dictionary compaction (#695 PR B) — the correctness core.

Four invariants, in order of how badly a regression would hurt:

* **I5** — no vocab ⇒ the remap is a literal early-return, so the encoder is
  BYTE-IDENTICAL to the pre-#695 code. Asserted by *object identity*, which is
  strictly stronger than numeric equality: it proves zero ops were added, so
  byte-identity cannot regress through a well-meaning refactor.

* **V4** — slice-init. Drawing the FULL ``(n_sae, d)`` table and keeping the allowed
  rows consumes the IDENTICAL RNG stream, so every downstream init (magnitude_embed,
  blocks, value_embeds, VE gates, cross-attention) is bitwise identical to the
  uncompacted run. This is what makes "byte-exact vs the reference" true on the
  MASKED path — i.e. on the default path.

* **V3** — a compacted model and a masked-but-uncompacted model produce BITWISE
  identical logits. Compaction is an exact identity, not a numerical trade.

* **Dead rows** — rows reachable only through masked/padding slots have zero effect on
  outputs and receive EXACTLY-zero gradient. This is what licenses deleting them, and
  what licenses mapping padding onto compact row 0.

NOTE on test hygiene: ``init_weights()`` zero-inits the encoder's ``attn.c_proj``,
which kills all gradient into q/k/v at init. A gradient assertion on a freshly
initialised model would therefore pass for the WRONG reason. Every gradient test here
runs on a de-degenerated model (all params randomised).
"""

from __future__ import annotations

import ast
import copy
import inspect
import random
import textwrap

import pytest
import torch as t
import torch._dynamo as dynamo
import torch.nn as nn

import nanochat.flash_attention as _fa_mod
from errors import FeatureVocabError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig, has_ve
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
    _average_row_norm,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
    feature_vocab_from_state_dict,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.eval.target_conditioning import SelfConditionedTarget
from steering.refined_types import PositiveInt

N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)  # V = 12
BLACKLIST = tuple(i for i in range(N_SAE) if i not in ALLOWED)
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
MASK = FeatureMask(mode=FeatureMaskMode.blacklist, ids=BLACKLIST)

# A vocab that EXCLUDES SAE id 0 — the whitelist case, where compact row 0 is a real
# feature that padding also lands on. Several traps only exist in this configuration.
NO_ZERO = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10, 20, 30))


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force the SDPA attention backend for every test in this module.

    WITHOUT THIS THE PROOFS BELOW DO NOT RUN — they crash. The decoder's self-attention routes
    to compiled FlexAttention, which cannot lower on CPU, so any full-model forward raises
    ``InductorError: IndexError: tuple index out of range`` from ``nanochat/flash_attention.py``.
    51 other test modules that forward an encdec on CPU install exactly this fixture; this one
    did not, and the crash was masked because every command used to verify this PR happened to
    export ``NANOCHAT_ATTN_IMPL=sdpa``. A proof that only passes under one person's shell is
    not a proof.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _enc_config(n_layer=2) -> FeatureEncoderConfig:
    return FeatureEncoderConfig(
        n_sae_features=N_SAE,
        num_buckets=4,
        n_embd=32,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )


def _encoder(vocab: FeatureVocab | None, seed: int = 0, n_layer: int = 2) -> FeatureEncoder:
    """Build exactly as production does: META device → to_empty → seed → init_weights.

    This ordering is LOAD-BEARING for slice-init byte-exactness, and it is subtle:
    ``nn.Embedding.reset_parameters()`` draws ``numel()`` randoms **at construction**,
    so on a REAL device a compact table draws fewer than a full one and the RNG streams
    diverge before ``init_weights()`` is even reached. Meta-device construction draws
    nothing, and the training script additionally reseeds immediately before the
    cold-start ``init_weights()`` call. See ``TestMetaConstructionContract``.
    """
    with t.device("meta"):
        enc = FeatureEncoder(_enc_config(n_layer), feature_vocab=vocab)
    enc.to_empty(device="cpu")
    t.manual_seed(seed)
    enc.init_weights()
    return enc


def _encoder_twice(vocab: FeatureVocab | None, seed: int = 0) -> FeatureEncoder:
    """Production's DOUBLE ``init_weights()``: seeded ONCE, then initialised twice.

    The second init continues the same global RNG stream — it is not re-seeded — so it is
    a genuine second draw. Both models must be built this way *independently* (as separate
    processes do), never interleaved, or they consume different stream offsets and any
    comparison between them measures the harness rather than the code.
    """
    with t.device("meta"):
        enc = FeatureEncoder(_enc_config(), feature_vocab=vocab)
    enc.to_empty(device="cpu")
    t.manual_seed(seed)
    enc.init_weights()
    enc.init_weights()
    return enc


def _model(
    vocab: FeatureVocab | None,
    seed: int = 0,
    fp: bool = False,
    *,
    fp_loss_mode: FeaturePredLossMode = FeaturePredLossMode.BEST_TOKEN,
    fp_loss_fn: FeaturePredLossFn = FeaturePredLossFn.BCE,
) -> EncoderDecoderGPT:
    dec = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc = EncoderConfig(n_embd=32, depth=2, n_head=None, n_kv_head=None, head_dim=16)
    xa = CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None)
    fp_cfg = (
        FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=fp_loss_mode,
            loss_fn=fp_loss_fn,
        )
        if fp
        else None
    )
    with t.device("meta"):
        m = EncoderDecoderGPT(
            dec,
            enc,
            xa,
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_pred_config=fp_cfg,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(seed)
    m.init_weights()
    return m


def _de_degenerate(module: t.nn.Module, seed: int = 1234) -> None:
    """init_weights() zero-inits attn.c_proj and ve_gate, which would make gradient
    assertions pass for the wrong reason. Give every parameter a nonzero value."""
    g = t.Generator().manual_seed(seed)
    with t.no_grad():
        for p in module.parameters():
            p.copy_(t.randn(p.shape, generator=g, dtype=t.float32) * 0.05)


def _batch(B=2, S=2, F=5, real_ids=(10, 20, 30)):
    """Real features from ALLOWED; the remaining slots are PADDING carrying id 0."""
    fids = t.zeros(B, S, F, dtype=t.int64)
    mags = t.zeros(B, S, F, dtype=t.int64)
    mask = t.zeros(B, S, F, dtype=t.bool)
    for b in range(B):
        for s in range(S):
            fids[b, s, : len(real_ids)] = t.tensor(real_ids)
            mags[b, s, : len(real_ids)] = t.tensor([1, 2, 3, 1, 2, 3][: len(real_ids)])
            mask[b, s, : len(real_ids)] = True
    assert (fids[~mask] == 0).all(), "padding must carry feature id 0 — that is the trap"
    return fids, mags, mask


def _feature_indexed_names(module: t.nn.Module) -> set[str]:
    """Parameter names of the SAE-feature-id-indexed tables, found by module IDENTITY.

    Deliberately not a name match: ``EncoderDecoderGPT`` has its own ``value_embeds``
    (TOKEN-indexed, ``vocab_size`` rows) alongside the encoder's, and a substring rule
    would row-subset the decoder's tables — which happen to be big enough that the bug
    would look like a plausible tensor rather than an error.
    """
    enc = module.encoder if hasattr(module, "encoder") else module
    tables = [enc.feature_embed, *(enc.value_embeds[k] for k in enc.value_embeds)]  # ty: ignore[not-iterable, not-subscriptable, unresolved-attribute]
    table_params = {id(p) for tbl in tables for p in tbl.parameters()}  # ty: ignore[unresolved-attribute]
    return {n for n, p in module.named_parameters() if id(p) in table_params}


def _sync_onto_compact(full: t.nn.Module, comp: t.nn.Module, vocab: FeatureVocab) -> None:
    """Copy ``full``'s parameters onto ``comp``, row-subsetting the feature-indexed ones.

    Makes the two models the SAME model expressed two ways, so any later difference is
    caused by the remap itself and not by their inits.
    """
    rows = t.tensor(vocab.original_ids)
    feature_indexed = _feature_indexed_names(full)
    assert feature_indexed, "no feature-indexed tables found — the sync would be a no-op"
    with t.no_grad():
        fd, cd = dict(full.named_parameters()), dict(comp.named_parameters())
        assert fd.keys() == cd.keys()
        for name, pf in fd.items():
            cd[name].copy_(pf[rows] if name in feature_indexed else pf)


def _synced_pair(vocab: FeatureVocab, seed: int = 0) -> tuple[FeatureEncoder, FeatureEncoder]:
    """(uncompacted, compacted) ENCODERS holding bitwise the same weights."""
    full, comp = _encoder(None, seed=seed), _encoder(vocab, seed=seed)
    _de_degenerate(full, seed=seed + 1000)
    _sync_onto_compact(full, comp, vocab)
    return full, comp


def _synced_models(
    vocab: FeatureVocab, seed: int = 0
) -> tuple[EncoderDecoderGPT, EncoderDecoderGPT]:
    """(masked-but-UNCOMPACTED, compacted) MODELS holding bitwise the same weights.

    The reference can no longer be produced by production config — compaction is
    always-on once a mask resolves — but it is constructible here because
    ``feature_vocab`` (a ctor arg that sizes the tables) is independent of
    ``feature_mask`` (a post-init attribute that gates selection).
    """
    full, comp = _model(None, seed=seed), _model(vocab, seed=seed)
    _de_degenerate(full, seed=seed + 1000)
    _sync_onto_compact(full, comp, vocab)
    full.feature_mask = MASK  # the uncompacted model still needs the selection gate
    return full, comp


# ===========================================================================
# I5 — no vocab ⇒ byte-identical
# ===========================================================================


class TestI5ByteIdentity:
    def test_no_vocab_means_the_remap_is_a_literal_early_return(self):
        """STRONGER than numeric equality: the SAME OBJECT comes back, proving zero
        ops were added. Byte-identity cannot regress through a refactor."""
        enc = _encoder(None)
        ids = t.tensor([[0, 3, 7]], dtype=t.int64)
        mask = t.ones(1, 3, dtype=t.bool)
        assert enc._to_compact(ids, mask) is ids

    def test_no_vocab_registers_NO_buffers(self):
        """A legacy checkpoint has no vocab keys; a fresh no-mask model must not add
        any, or every existing strict=True load breaks."""
        sd = _encoder(None).state_dict()
        assert VOCAB_IDS_KEY.removeprefix("encoder.") not in sd
        assert VOCAB_N_SAE_KEY.removeprefix("encoder.") not in sd

    def test_no_vocab_table_is_full_width(self):
        enc = _encoder(None)
        assert enc.feature_embed.weight.shape[0] == N_SAE
        for k in enc.value_embeds:
            assert enc.value_embeds[k].weight.shape[0] == N_SAE  # ty: ignore[not-subscriptable]

    def test_no_vocab_init_consumes_the_reference_rng_stream(self):
        """Two no-vocab encoders from the same seed are bitwise identical (a baseline
        that would catch an accidental extra/removed RNG draw)."""
        a, b = _encoder(None, seed=7), _encoder(None, seed=7)
        for (na, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
            assert t.equal(pa, pb), na


# ===========================================================================
# V4 — slice-init: the compacted model IS the reference, row-for-row
# ===========================================================================


class TestV4SliceInit:
    def test_compact_feature_embed_rows_are_BITWISE_the_full_tables_allowed_rows(self):
        full, comp = _encoder(None, seed=42), _encoder(VOCAB, seed=42)
        expected = full.feature_embed.weight[t.tensor(ALLOWED)]
        assert t.equal(comp.feature_embed.weight, expected)

    def test_compact_value_embeds_rows_are_BITWISE_the_full_tables_allowed_rows(self):
        full, comp = _encoder(None, seed=42), _encoder(VOCAB, seed=42)
        for k in comp.value_embeds:
            expected = full.value_embeds[k].weight[t.tensor(ALLOWED)]  # ty: ignore[not-subscriptable]
            assert t.equal(comp.value_embeds[k].weight, expected), f"value_embeds[{k}]"  # ty: ignore[invalid-argument-type]

    def test_EVERY_other_parameter_is_bitwise_identical(self):
        """THE point of slice-init. Drawing the FULL table keeps the global RNG stream
        aligned, so magnitude_embed, the blocks, the VE gates — everything downstream —
        matches the uncompacted run exactly. Without slice-init these all diverge."""
        full, comp = _encoder(None, seed=42), _encoder(VOCAB, seed=42)
        fd, cd = dict(full.named_parameters()), dict(comp.named_parameters())
        assert fd.keys() == cd.keys()
        for name, full_param in fd.items():
            if "feature_embed" in name or "value_embeds" in name:
                continue  # feature-indexed: compared above
            assert t.equal(full_param, cd[name]), f"{name} diverged — RNG stream broke"

    def test_rng_stream_is_left_in_the_identical_state_after_init(self):
        """Anything initialised AFTER the encoder (cross-attention, in the full model)
        must also match, which requires the RNG to be at the same offset."""
        t.manual_seed(99)
        _encoder(None)  # noqa: F841 -- consumes the stream
        after_full = t.randn(8)
        t.manual_seed(99)
        _encoder(VOCAB)
        after_compact = t.randn(8)
        assert t.equal(after_full, after_compact)

    def test_whole_model_slice_init_leaves_the_decoder_bitwise_identical(self):
        full, comp = _model(None, seed=5), _model(VOCAB, seed=5)
        fd, cd = dict(full.named_parameters()), dict(comp.named_parameters())
        for name, full_param in fd.items():
            if "encoder.feature_embed" in name or "encoder.value_embeds" in name:
                continue
            assert t.equal(full_param, cd[name]), f"{name} diverged"


class TestMetaConstructionContract:
    """Slice-init byte-exactness has a PRECONDITION that is easy to violate silently.

    ``nn.Embedding.reset_parameters()`` draws ``numel()`` randoms at CONSTRUCTION. On a
    real device a compact table therefore draws fewer randoms than a full one, and the
    global RNG stream diverges *before* ``init_weights()`` runs — silently destroying
    byte-exactness no matter how correct slice-init itself is.

    Production is safe two ways: it constructs on the ``meta`` device (which draws
    nothing), and it calls ``manual_seed`` AFTER construction. These tests pin both
    facts so a future refactor that constructs on a real device fails loudly here
    rather than quietly ruining the byte-identity guarantee.
    """

    def test_meta_construction_consumes_NO_rng(self):
        t.manual_seed(0)
        with t.device("meta"):
            FeatureEncoder(_enc_config(), feature_vocab=None)
        after_full = t.randn(4)

        t.manual_seed(0)
        with t.device("meta"):
            FeatureEncoder(_enc_config(), feature_vocab=VOCAB)
        after_compact = t.randn(4)

        assert t.equal(after_full, after_compact), (
            "meta construction must not consume RNG, or slice-init cannot be byte-exact"
        )

    def test_REAL_device_construction_DOES_consume_rng_differentially(self):
        """The trap, made explicit. This is why _encoder() builds on meta.

        If this ever starts passing, nn.Embedding stopped initialising at construction
        and the meta-device dance is no longer load-bearing — but until then, seeding
        BEFORE constructing a compacted encoder on a real device silently breaks V4.
        """
        t.manual_seed(0)
        FeatureEncoder(_enc_config(), feature_vocab=None)
        after_full = t.randn(4)

        t.manual_seed(0)
        FeatureEncoder(_enc_config(), feature_vocab=VOCAB)
        after_compact = t.randn(4)

        assert not t.equal(after_full, after_compact), (
            "expected real-device construction to draw a DIFFERENT number of randoms "
            "for a compact table — if this changed, update the byte-exactness docs"
        )


# ===========================================================================
# V3 — compacted logits == masked-but-uncompacted logits, BITWISE
# ===========================================================================


class TestV3CompactionEquality:
    def test_encoder_output_is_bitwise_identical(self):
        full, comp = _encoder(None, seed=3), _encoder(VOCAB, seed=3)
        _de_degenerate(full)
        # re-derive the compact tables from the de-degenerated full ones
        with t.no_grad():
            comp.feature_embed.weight.copy_(full.feature_embed.weight[t.tensor(ALLOWED)])
            for k in comp.value_embeds:
                comp.value_embeds[k].weight.copy_(full.value_embeds[k].weight[t.tensor(ALLOWED)])  # ty: ignore[call-non-callable, not-subscriptable]
            for (_, pf), (nc, pc) in zip(
                full.named_parameters(), comp.named_parameters(), strict=True
            ):
                if "feature_embed" not in nc and "value_embeds" not in nc:
                    pc.copy_(pf)
        fids, mags, mask = _batch()
        with t.no_grad():
            out_full = full(fids, mags, mask)
            out_comp = comp(fids, mags, mask)
        assert t.equal(out_full.layout.valid_mask, out_comp.layout.valid_mask)
        assert t.equal(out_full.layout.flat_indices, out_comp.layout.flat_indices)
        assert t.equal(out_full.values, out_comp.values), (
            "compaction is supposed to be an IDENTITY"
        )

    def test_full_model_logits_are_bitwise_identical(self):
        """The headline equality of issue #695.

        Note the reference model (masked but UNCOMPACTED) can no longer be produced by
        production config — compaction is always-on when a mask resolves. It is
        constructible here because feature_vocab (a ctor arg that sizes the tables) is
        independent of feature_mask (a post-init attribute that gates selection).
        """
        full, comp = _model(None, seed=11), _model(VOCAB, seed=11)
        _de_degenerate(full)
        with t.no_grad():
            for (nf, pf), (nc, pc) in zip(
                full.named_parameters(), comp.named_parameters(), strict=True
            ):
                assert nf == nc
                if (
                    "encoder.feature_embed" in nc
                    or "encoder.value_embeds" in nc
                ):
                    pc.copy_(pf[t.tensor(ALLOWED)])
                else:
                    pc.copy_(pf)
        full.eval()
        comp.eval()
        full.feature_mask = MASK  # the uncompacted model still needs the selection gate

        idx = t.randint(0, 128, (2, 16))
        seg_map = (t.arange(16) // 8).unsqueeze(0).expand(2, 16).contiguous()
        fids, mags, mask = _batch(B=2, S=2)
        with t.no_grad():
            lg_full = full(idx, None, fids, mags, mask, seg_map)
            lg_comp = comp(idx, None, fids, mags, mask, seg_map)
        assert t.equal(lg_full, lg_comp), "compacted logits must be BITWISE identical"


# ===========================================================================
# Dead rows: padding is provably harmless
# ===========================================================================


class TestPaddingAndDeadRows:
    def test_out_of_vocab_id_0_in_PADDING_does_not_raise(self):
        """Feature id 0 IS out of this vocab... no wait — 0 IS in ALLOWED. Use a vocab
        that excludes 0, which is the real-world whitelist case."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10))
        enc = _encoder(vocab)
        fids, mags, mask = _batch(F=5, real_ids=(3, 7, 10))
        assert (fids[~mask] == 0).all()
        assert 0 not in vocab.original_ids, "id 0 must be OUT of vocab for this test"
        enc(fids, mags, mask)  # must NOT raise

    def test_padding_maps_to_compact_row_zero(self):
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10))
        enc = _encoder(vocab)
        ids = t.tensor([[3, 0, 0]], dtype=t.int64)
        mask = t.tensor([[True, False, False]])
        got = enc._to_compact(ids, mask)
        assert got.tolist() == [[0, 0, 0]], "padding must land on a VALID row"

    def test_compact_row_for_padding_receives_exactly_zero_gradient(self):
        """The proof that licenses mapping padding anywhere: masked encoder positions
        are masked as KEYS in cross-attention, so they contribute exactly 0."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10, 20, 30))
        m = _model(vocab, seed=8)
        _de_degenerate(m)
        m.train()
        idx = t.randint(0, 128, (2, 16))
        tgt = t.randint(0, 128, (2, 16))
        seg_map = (t.arange(16) // 8).unsqueeze(0).expand(2, 16).contiguous()
        fids, mags, mask = _batch(B=2, S=2, real_ids=(10, 20, 30))
        m.zero_grad(set_to_none=True)
        m(idx, tgt, fids, mags, mask, seg_map).backward()

        g = m.encoder.feature_embed.weight.grad
        used_original = {10, 20, 30}
        used_compact = {vocab.original_ids.index(i) for i in used_original}
        nonzero = {int(i) for i in (g.abs().sum(1) > 0).nonzero().flatten().tolist()}  # ty: ignore[unresolved-attribute]
        assert nonzero == used_compact, "exactly the rows of the REAL features may receive gradient"
        # compact row 0 == original id 3, which appears ONLY via padding
        assert g[0].abs().sum().item() == 0.0  # ty: ignore[not-subscriptable]

    def test_all_masked_row_does_not_raise(self):
        """build_no_feature_inputs emits an all-masked (1,1,1) input carrying id 0.
        Every slot is padding, so it must never trip the vocab guard."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10))
        enc = _encoder(vocab)
        enc(
            t.zeros(1, 1, 1, dtype=t.int64),
            t.zeros(1, 1, 1, dtype=t.int64),
            t.zeros(1, 1, 1, dtype=t.bool),
        )

    def test_fp8_internal_pad_reinjects_id_zero_and_is_still_safe(self):
        """The aligned packed tail reintroduces id zero after caller-side scrubbing."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10))
        enc = _encoder(vocab)
        F_unaligned = 3
        assert F_unaligned % FP8_ALIGNMENT != 0
        fids = t.tensor([[[3, 7, 10]]], dtype=t.int64)
        mags = t.zeros(1, 1, 3, dtype=t.int64)
        mask = t.ones(1, 1, 3, dtype=t.bool)
        out = enc(fids, mags, mask)  # must not raise despite the injected 0s
        assert out.values.shape[1] % (2 * FP8_ALIGNMENT) == 0
        assert out.layout.valid_mask[:3].all()
        assert not out.layout.valid_mask[3:].any(), "packed tail slots must be invalid"


# ===========================================================================
# Out-of-vocab: fails loud in EAGER and COMPILED, across every dynamic mode
# ===========================================================================


class TestOutOfVocabGuard:
    def test_real_out_of_vocab_id_raises_in_eager(self):
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        fids[0, 0, 0] = 5  # 5 is NOT in ALLOWED, and this slot is mask=True
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc(fids, mags, mask)

    def test_the_message_is_actionable(self):
        """No host sync ⇒ we cannot name the offending ids, so the message MUST name
        the shape of the problem and the remedy instead."""
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        fids[0, 0, 0] = 5
        with pytest.raises(RuntimeError) as exc:
            enc(fids, mags, mask)
        msg = str(exc.value)
        assert "12" in msg and "64" in msg, "must name V and n_sae"
        assert "dump_feature_mask" in msg, "must name the remedy"

    def test_the_message_fits_cuda_async_assert_limit(self):
        enc = _encoder(VOCAB)
        encoded = enc._vocab_oov_msg.encode("utf-8")
        assert len(encoded) < 255

    @pytest.mark.slow
    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
    def test_valid_compact_forward_does_not_fail_on_cuda_message_validation(self):
        """CUDA validates the async-assert message even when all IDs are valid."""
        enc = _encoder(VOCAB).cuda()
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        enc(fids.cuda(), mags.cuda(), mask.cuda())
        t.cuda.synchronize()

    @pytest.mark.parametrize("dynamic", [False, True, None])
    def test_out_of_vocab_raises_UNDER_COMPILE_in_every_dynamic_mode(self, dynamic):
        """THE test that catches the silent last-row wraparound.

        nn.Embedding(-1) raises in eager but SILENTLY RETURNS THE LAST ROW when
        compiled. Without torch._assert_async the compiled path is silently wrong.
        resolve_compile_dynamic can select any of these three modes in production.
        """
        dynamo.reset()
        enc = _encoder(VOCAB)
        c = t.compile(enc._to_compact, fullgraph=True, dynamic=dynamic)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        ids_flat = fids.reshape(-1, fids.shape[-1])
        mask_flat = mask.reshape_as(ids_flat)
        c(ids_flat, mask_flat)  # valid ids: no false positive

        bad = ids_flat.clone()
        bad[0, 0] = 5
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            c(bad, mask_flat)

    def test_compiles_fullgraph_with_zero_graph_breaks(self):
        dynamo.reset()
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch()
        ids_flat = fids.reshape(-1, fids.shape[-1])
        mask_flat = mask.reshape_as(ids_flat)
        explanation = dynamo.explain(enc._to_compact)(ids_flat, mask_flat)
        assert explanation.graph_break_count == 0

    def test_id_beyond_n_sae_raises(self):
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        fids[0, 0, 0] = N_SAE + 5
        with pytest.raises(RuntimeError):
            enc(fids, mags, mask)

    def test_hot_path_performs_NO_host_sync(self):
        """The owner chose sync-free over rich diagnostics. That property regresses
        INVISIBLY, so pin it: _to_compact must never call .item()/bool()/.cpu()."""
        src = inspect.getsource(FeatureEncoder._to_compact)
        for banned in (".item()", "bool(", ".cpu()", ".tolist()", ".numpy()"):
            assert banned not in src, f"_to_compact must not host-sync: found {banned!r}"


class _Wrapper(nn.Module):
    """A plain module holding the model as a CHILD — an EMA/LoRA/compile-style wrapper."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner


class TestNoGradDecoratorsAreNotSILENTLYSTOLEN:
    """A decorator binds to whatever function FOLLOWS it. Insert a new method directly beneath
    ``@torch.no_grad()`` and it silently rebinds — ``init_buffers`` becomes undecorated, the new
    method becomes decorated, and a line diff shows **nothing**, because the decorator line is
    unchanged context.

    That is exactly what happened: a ``to_empty`` override was inserted under
    ``EncoderDecoderGPT``'s ``@torch.no_grad()``, and ``init_buffers`` — a public entry point on
    every load path whose whole job is writing buffers on a live model — ran with autograd
    enabled for the rest of the PR. Only an AST comparison could see it.
    """

    @pytest.mark.parametrize(
        ("cls", "method"),
        [
            (EncoderDecoderGPT, "init_weights"),
            (EncoderDecoderGPT, "init_buffers"),
            (FeatureEncoder, "init_vocab_buffers"),
        ],
    )
    def test_the_buffer_writing_method_still_carries_no_grad(self, cls, method: str) -> None:
        decorators = _decorators_of(cls, method)
        module = inspect.getmodule(cls)
        assert module is not None
        torch_names = {name for name, value in vars(module).items() if value is t}
        assert any(f"{name}.no_grad()" in decorators for name in torch_names), (
            f"{cls.__name__}.{method} lost its @torch.no_grad() — check whether a method was "
            f"inserted directly beneath the decorator, which rebinds it. Found: {decorators}"
        )


def _decorators_of(cls: type, method: str) -> list[str]:
    """The decorators on ``cls.method``, read from SOURCE.

    Source, not ``__wrapped__``: the bug is that the decorator binds to the WRONG function, and
    a runtime check on the right function would happily pass while the wrong one is decorated.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            return [ast.unparse(d) for d in node.decorator_list]
    raise AssertionError(f"{cls.__name__}.{method} not found — the test is stale")


class TestBuffers:
    def test_compacted_encoder_registers_both_persistent_buffers(self):
        """BOTH ``feature_vocab_ids`` and ``feature_vocab_n_sae`` are persistent — they are
        the WHOLE of the vocab state (``_to_compact`` derives the remap from ``ids`` per
        forward), so ``load_state_dict`` must restore them."""
        sd = _encoder(VOCAB).state_dict()
        assert VOCAB_IDS_KEY.removeprefix("encoder.") in sd
        assert VOCAB_N_SAE_KEY.removeprefix("encoder.") in sd

    def test_a_no_mask_encoder_registers_NEITHER_vocab_buffer(self):
        """A legacy/uncompacted checkpoint carries no vocab keys; a no-mask encoder must
        add neither buffer, or every existing strict=True load of a full-width model breaks.
        """
        enc = _encoder(None)
        sd = enc.state_dict()
        assert VOCAB_IDS_KEY.removeprefix("encoder.") not in sd
        assert VOCAB_N_SAE_KEY.removeprefix("encoder.") not in sd
        assert not hasattr(enc, "feature_vocab_ids")
        assert not hasattr(enc, "feature_vocab_n_sae")

    def test_buffer_contents(self):
        enc = _encoder(VOCAB)
        assert enc.feature_vocab_ids.tolist() == list(ALLOWED)
        assert int(enc.feature_vocab_n_sae.item()) == N_SAE

    def test_tables_are_sized_V_not_n_sae(self):
        enc = _encoder(VOCAB)
        assert enc.feature_embed.weight.shape[0] == len(ALLOWED)
        for k in enc.value_embeds:
            assert enc.value_embeds[k].weight.shape[0] == len(ALLOWED)  # ty: ignore[not-subscriptable]

    def test_config_n_sae_features_STAYS_the_original_width(self):
        """eval_adapter, target_conditioning, model_detection and harmfulness_eval all
        read this to bounds-check ORIGINAL ids. Redefining it to V corrupts all of them."""
        enc = _encoder(VOCAB)
        assert enc.config.n_sae_features == N_SAE


# ===========================================================================
# Model-level guards
# ===========================================================================


class TestModelGuards:
    def test_fp_head_plus_compaction_uses_the_effective_width(self):
        model = _model(VOCAB, fp=True)
        assert model.feature_pred_head is not None
        assert model.feature_pred_head.out_features == len(ALLOWED)
        assert model.feature_prediction_width == len(ALLOWED)
        assert model._n_sae_features == N_SAE

    def test_fp_head_without_compaction_is_fine(self):
        m = _model(None, fp=True)
        assert m.feature_pred_head is not None

    def test_compact_fp_init_is_the_exact_full_row_slice_and_preserves_rng(self):
        full = _model(None, seed=2026, fp=True)
        full_rng = t.get_rng_state().clone()
        compact = _model(VOCAB, seed=2026, fp=True)
        compact_rng = t.get_rng_state().clone()

        assert full.feature_pred_head is not None
        assert compact.feature_pred_head is not None
        rows = t.tensor(ALLOWED, dtype=t.int64)
        assert t.equal(
            compact.feature_pred_head.weight,
            full.feature_pred_head.weight[rows],
        )
        assert t.equal(
            compact.feature_pred_head.bias,
            full.feature_pred_head.bias[rows],
        )
        assert t.equal(compact_rng, full_rng)

    def test_compact_fp_loss_and_head_gradients_match_the_full_masked_reference(self):
        full = _model(None, seed=2030, fp=True)
        compact = _model(VOCAB, seed=2030, fp=True)
        full.feature_mask = MASK
        compact.feature_mask = MASK

        fids, mags, mask = _batch(B=1, S=2)
        idx = t.tensor([[1, 2, 3, 4]], dtype=t.int64)
        targets = t.tensor([[2, 3, 4, 5]], dtype=t.int64)
        seg_map = t.tensor([[0, 0, 1, 1]], dtype=t.int64)
        seg_doc_id = t.tensor([[0, 0]], dtype=t.int64)
        full_targets = t.zeros(1, 2, N_SAE, dtype=t.float32)
        full_targets[0, 1, t.tensor(ALLOWED[:4])] = t.tensor(
            [0.2, 0.4, 0.6, 0.8]
        )
        compact_targets = full_targets[..., t.tensor(ALLOWED)]

        full_result = full(
            idx,
            targets,
            fids,
            mags,
            mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=full_targets,
        )
        compact_result = compact(
            idx,
            targets,
            fids,
            mags,
            mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=compact_targets,
        )
        assert isinstance(full_result, ForwardResult)
        assert isinstance(compact_result, ForwardResult)
        assert t.equal(full_result.loss, compact_result.loss)
        assert t.equal(full_result.feature_pred_loss, compact_result.feature_pred_loss)

        full_result.loss.backward()
        compact_result.loss.backward()
        assert full.feature_pred_head is not None
        assert compact.feature_pred_head is not None
        rows = t.tensor(ALLOWED, dtype=t.int64)
        assert t.equal(
            compact.feature_pred_head.weight.grad,  # ty: ignore[invalid-argument-type]
            full.feature_pred_head.weight.grad[rows],  # ty: ignore[not-subscriptable]
        )
        assert t.equal(
            compact.feature_pred_head.bias.grad,  # ty: ignore[invalid-argument-type]
            full.feature_pred_head.bias.grad[rows],  # ty: ignore[not-subscriptable]
        )

    @pytest.mark.parametrize(
        "loss_fn",
        [FeaturePredLossFn.BCE, FeaturePredLossFn.MSE],
    )
    def test_compact_fp_training_forward_and_backward_matches_compiled_eager_backend(
        self,
        loss_fn: FeaturePredLossFn,
    ):
        eager = _model(
            VOCAB,
            seed=2031,
            fp=True,
            fp_loss_mode=FeaturePredLossMode.MAX_POOL,
            fp_loss_fn=loss_fn,
        )
        compiled_model = _model(
            VOCAB,
            seed=2031,
            fp=True,
            fp_loss_mode=FeaturePredLossMode.MAX_POOL,
            fp_loss_fn=loss_fn,
        )
        eager.feature_mask = MASK
        compiled_model.feature_mask = MASK
        compiled = t.compile(compiled_model, backend="eager")
        fids, mags, feature_mask = _batch(B=1, S=2)
        idx = t.tensor([[1, 2, 3, 4]], dtype=t.int64)
        targets = t.tensor([[2, 3, 4, 5]], dtype=t.int64)
        seg_map = t.tensor([[0, 0, 1, 1]], dtype=t.int64)
        seg_doc_id = t.tensor([[0, 0]], dtype=t.int64)
        fp_targets = t.zeros(1, 2, VOCAB.size, dtype=t.float32)
        fp_targets[0, 1, :4] = t.tensor([0.2, 0.4, 0.6, 0.8])

        eager_result = eager(
            idx,
            targets,
            fids,
            mags,
            feature_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )
        compiled_result = compiled(
            idx,
            targets,
            fids,
            mags,
            feature_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )
        assert isinstance(eager_result, ForwardResult)
        assert isinstance(compiled_result, ForwardResult)
        assert t.equal(compiled_result.loss, eager_result.loss)
        assert t.equal(
            compiled_result.feature_pred_loss,
            eager_result.feature_pred_loss,
        )
        eager_result.loss.backward()
        compiled_result.loss.backward()
        assert eager.feature_pred_head is not None
        assert compiled_model.feature_pred_head is not None
        assert eager.feature_pred_head.weight.grad is not None
        assert compiled_model.feature_pred_head.weight.grad is not None
        assert t.equal(
            compiled_model.feature_pred_head.weight.grad,
            eager.feature_pred_head.weight.grad,
        )

    def test_compaction_without_fp_head_is_fine(self):
        m = _model(VOCAB, fp=False)
        assert m.feature_pred_head is None
        assert m.feature_vocab == VOCAB

    def test_model_exposes_the_vocab_and_its_size(self):
        m = _model(VOCAB)
        assert m.feature_vocab == VOCAB
        assert m.feature_vocab_size == len(ALLOWED)
        assert _model(None).feature_vocab_size == N_SAE

    def test_the_FEATURE_table_height_is_NOT_called_vocab_size(self):
        """FINDING 11 — a name collision with teeth.

        ``vocab_size`` means the TOKEN vocabulary everywhere else in this repo
        (``GPTConfig.vocab_size == 32768``; this very model slices its logits with
        ``logits[..., : self.dec_config.vocab_size]``). A property of that name on the
        MODEL returning the encoder's FEATURE-table height is a trap: any generic
        consumer reaching for the near-universal ``model.vocab_size`` idiom — a sampler,
        a logit slicer, an HF-style adapter — would silently get V and slice against the
        wrong width. There is no error to catch; the logits just come back truncated.

        So the attribute must NOT exist under that name. Deleting it is not enough if
        someone re-adds it later, which is what this pins.
        """
        m = _model(VOCAB)
        assert not hasattr(m, "vocab_size"), (
            "EncoderDecoderGPT must not expose `vocab_size` — it collides with the TOKEN "
            "vocab (dec_config.vocab_size). Use `feature_vocab_size` for the encoder's "
            "feature-table height."
        )
        # ...and the token vocab is still reachable where it always was.
        assert m.dec_config.vocab_size == 128

    def test_param_count_drops_by_exactly_the_dead_rows(self):
        full, comp = _model(None, seed=1), _model(VOCAB, seed=1)
        d_full = full.num_scaling_params()
        d_comp = comp.num_scaling_params()
        enc_cfg = full.encoder.config
        kv = enc_cfg.n_kv_head * enc_cfg.head_dim
        n_ve = len(full.encoder.value_embeds)
        dropped_rows = N_SAE - len(ALLOWED)
        expected = dropped_rows * enc_cfg.n_embd + n_ve * dropped_rows * kv
        assert d_full["encoder"] - d_comp["encoder"] == expected
        assert d_full["total"] - d_comp["total"] == expected

    def test_setup_optimizer_still_accounts_for_every_parameter(self):
        """model.py's exact param-count assert must stay green: buffers are not params."""
        _model(VOCAB).setup_optimizer()

    def test_assert_vocab_consistent_catches_an_OVERWRITTEN_ids_buffer(self):
        """``load_state_dict(assign=True)`` REPLACES the persistent ``feature_vocab_ids``
        buffer. If the checkpoint describes a DIFFERENT vocabulary than the model was
        configured with, every embedding row is silently reinterpreted — and since
        ``_to_compact`` derives its remap from this very buffer, ``ids`` matching the
        configured vocab is the WHOLE invariant. This is the guard that turns the swap
        into a raise."""
        m = _model(VOCAB)
        m.assert_vocab_consistent()
        with t.no_grad():
            m.encoder.feature_vocab_ids[0] = 99  # a different vocabulary
        with pytest.raises(FeatureVocabError, match="feature_vocab_ids"):
            m.assert_vocab_consistent()

    def test_assert_vocab_consistent_is_DTYPE_STRICT_about_the_ids_buffer(self):
        """A real hole that was just closed: ``torch.equal`` reports an int32 and an int64
        buffer holding the SAME values as EQUAL, so a checkpoint whose ``feature_vocab_ids``
        is int32 would slip past a value-only check. The guard compares dtype explicitly, so
        an int32 ids buffer RAISES even though its values match."""
        m = _model(VOCAB)
        with t.no_grad():
            m.encoder.feature_vocab_ids = m.encoder.feature_vocab_ids.to(t.int32)
        assert m.encoder.feature_vocab_ids.tolist() == list(ALLOWED), (
            "the fixture must keep the VALUES identical, so only the dtype differs"
        )
        with pytest.raises(FeatureVocabError, match="feature_vocab_ids"):
            m.assert_vocab_consistent()

    def test_state_dict_round_trip_preserves_logits(self):
        m = _model(VOCAB, seed=17)
        _de_degenerate(m)
        m.eval()
        idx = t.randint(0, 128, (2, 16))
        seg_map = (t.arange(16) // 8).unsqueeze(0).expand(2, 16).contiguous()
        fids, mags, mask = _batch(B=2, S=2)
        with t.no_grad():
            before = m(idx, None, fids, mags, mask, seg_map)

        sd = m.state_dict()
        m2 = _model(VOCAB, seed=99)  # different init
        m2.load_state_dict(sd, strict=True)
        m2.assert_vocab_consistent()
        m2.eval()
        with t.no_grad():
            after = m2(idx, None, fids, mags, mask, seg_map)
        assert t.equal(before, after)

    def test_assert_vocab_consistent_catches_a_corrupted_n_sae_buffer(self):
        m = _model(VOCAB)
        with t.no_grad():
            m.encoder.feature_vocab_n_sae.fill_(N_SAE + 1)
        with pytest.raises(FeatureVocabError, match="n_sae"):
            m.assert_vocab_consistent()

    def test_assert_vocab_consistent_rejects_vocab_buffers_WITHOUT_a_FeatureVocab(self):
        """An uncompacted model carrying vocab buffers is incoherent state: the tables are
        full-width but something claims a vocab. Fail rather than guess which is right."""
        m = _model(None)
        m.assert_vocab_consistent()  # clean
        m.encoder.register_buffer(
            "feature_vocab_ids", t.tensor([0, 1], dtype=t.int64), persistent=True
        )
        with pytest.raises(FeatureVocabError, match="no FeatureVocab"):
            m.assert_vocab_consistent()


# ===========================================================================
# FINDING 8 — an IDENTITY vocab must not survive construction
# ===========================================================================


class TestAnIdentityVocabIsRefusedAtConstruction:
    """``EncoderDecoderGPT.__init__`` accepted an IDENTITY ``FeatureVocab`` (V == n_sae).

    It then registered the two vocab buffers — producing a checkpoint that **every load
    path refuses**: ``feature_vocab_from_state_dict`` RAISES on identity buffers, by
    design (no writer is supposed to be able to emit them). So you could build a model,
    save it, and find that nothing in the repo could ever read it back.

    ``FeatureVocab.from_mask`` normalizes an identity to ``None``, so no *config* path
    reaches here — but the constructor is public, and "unreachable today" is exactly the
    reasoning that lets a trap sit armed until someone walks into it. The module's
    load-bearing invariant is that a **non-``None`` FeatureVocab ALWAYS means genuinely
    compacted**; the constructor must enforce it, at the earliest point, rather than
    deferring to a load that can no longer happen.
    """

    IDENTITY = FeatureVocab(n_sae_features=N_SAE, original_ids=tuple(range(N_SAE)))

    def test_the_constructor_REFUSES_an_identity_vocab(self):
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            _model(self.IDENTITY)

    def test_the_BARE_ENCODER_refuses_it_too(self):
        """The guard has to live on the class that owns the tables, not on the wrapper.

        ``EncoderDecoderGPT`` refused an identity vocab while ``FeatureEncoder`` — the class
        that actually sizes ``feature_embed`` / ``value_embeds`` and registers the vocab
        buffers — accepted one. So the invariant held only for callers who came in through the
        model. Construct an encoder directly (tests do; future code could) and you still got
        the un-loadable checkpoint, from the object whose invariant it is.
        """
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            FeatureEncoder(_enc_config(), feature_vocab=self.IDENTITY)

    def test_the_message_names_the_invariant_AND_the_fix(self):
        with pytest.raises(FeatureVocabError) as exc:
            _model(self.IDENTITY)
        msg = str(exc.value)
        low = msg.lower()
        assert str(N_SAE) in msg, "must quote the shape it is refusing"
        assert "none" in low, "the fix is to pass None — say so"
        assert "compact" in low, "must name the invariant it is defending"

    def test_construction_and_LOADING_now_AGREE_about_an_identity(self):
        """The teeth. These two must not disagree: a constructor that accepts what every
        loader refuses is a machine for building unloadable checkpoints."""
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            _model(self.IDENTITY)

        # The state dict such a model WOULD have written. The loader refuses it — which is
        # precisely why the constructor above has to.
        with pytest.raises(FeatureVocabError, match="(?i)identity"):
            feature_vocab_from_state_dict(
                {
                    FEATURE_EMBED_KEY: t.zeros(N_SAE, 8),
                    VOCAB_IDS_KEY: t.arange(N_SAE, dtype=t.int64),
                    VOCAB_N_SAE_KEY: t.tensor(N_SAE, dtype=t.int64),
                }
            )

    def test_a_GENUINELY_compacted_vocab_still_builds(self):
        """The guard must narrow the door, not close it."""
        assert _model(VOCAB).feature_vocab == VOCAB

    def test_None_still_builds_the_full_width_model(self):
        m = _model(None)
        assert m.feature_vocab is None
        assert m.encoder.feature_embed.weight.shape[0] == N_SAE

    def test_the_NEAR_identity_V_equals_n_sae_minus_ONE_is_ACCEPTED(self):
        """The boundary. Dropping exactly one feature IS a compaction and must build."""
        near = FeatureVocab(
            n_sae_features=N_SAE, original_ids=tuple(i for i in range(N_SAE) if i != 5)
        )
        m = _model(near)
        assert m.feature_vocab == near
        assert m.encoder.feature_embed.weight.shape[0] == N_SAE - 1

    def test_from_mask_STILL_normalizes_an_all_allowing_mask_to_None(self):
        """The reason no *config* path can reach the guard — and I5 depends on it: an
        all-allowing mask must still build a full-width, byte-identical encoder.

        The all-allowing mask is a WHITELIST of every id. It is not, as ``feature_vocab``'s
        module docstring claims, ``{mode: blacklist, ids: []}`` — ``FeatureMask`` rejects an
        empty id set outright (``"feature_mask resolved to an empty set"``), so that config
        cannot be constructed at all.
        """
        allow_all = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(range(N_SAE)))
        assert FeatureVocab.from_mask(allow_all, N_SAE) is None
        assert _model(FeatureVocab.from_mask(allow_all, N_SAE)).feature_vocab is None


# ===========================================================================
# _to_compact under adversarial inputs
# ===========================================================================


class TestToCompactAdversarial:
    """The remap is three lines of tensor ops with no host sync. Every one of these
    inputs reaches it in production, and none of them can be diagnosed after the fact."""

    def test_duplicate_ids_in_one_segment_all_map_to_the_SAME_row(self):
        enc = _encoder(VOCAB)
        got = enc._to_compact(t.tensor([[10, 10, 10]], dtype=t.int64), t.ones(1, 3, dtype=t.bool))
        j = ALLOWED.index(10)
        assert got.tolist() == [[j, j, j]]

    def test_UNSORTED_ids_map_by_RANK_not_by_position(self):
        """compact id is a property of the ID, never of the slot it arrived in. A stray
        ``.sort()`` in the remap would pass every all-ascending test and die here."""
        enc = _encoder(VOCAB)
        got = enc._to_compact(t.tensor([[63, 0, 30, 7]], dtype=t.int64), t.ones(1, 4, dtype=t.bool))
        assert got.tolist() == [[ALLOWED.index(i) for i in (63, 0, 30, 7)]]
        assert got.tolist() == [[11, 0, 7, 2]]

    def test_id_EXACTLY_n_sae_raises_the_off_by_one_boundary(self):
        """``test_id_beyond_n_sae_raises`` uses n_sae + 5. The interesting id is n_sae
        itself — one past the largest legal id, and the value a fencepost bug produces."""
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        fids[0, 0, 0] = N_SAE
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc(fids, mags, mask)

    def test_the_LARGEST_LEGAL_id_is_accepted_when_it_is_in_the_vocab(self):
        """The mirror of the above: n_sae - 1 IS legal (and 63 is in ALLOWED). The range
        check must not be an off-by-one that rejects the last real feature."""
        enc = _encoder(VOCAB)
        assert N_SAE - 1 in ALLOWED
        got = enc._to_compact(t.tensor([[N_SAE - 1]], dtype=t.int64), t.ones(1, 1, dtype=t.bool))
        assert got.tolist() == [[ALLOWED.index(N_SAE - 1)]]

    def test_NEGATIVE_id_in_a_REAL_slot_raises(self):
        enc = _encoder(VOCAB)
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc._to_compact(t.tensor([[-1, 10]], dtype=t.int64), t.ones(1, 2, dtype=t.bool))

    def test_NEGATIVE_id_in_a_PADDING_slot_ALSO_raises(self):
        """CHARACTERISATION — and deliberately stricter than the docstring implies.

        Padding is otherwise waved through (``mask=False`` ⇒ any in-range id is mapped to
        row 0 without complaint), but the ``in_range`` term of the guard is NOT
        mask-gated, so a negative id raises even in a padding slot.

        That is the right call — every producer pads with a literal ``0``, so a negative
        padding value means a producer changed its padding convention (``-1`` is the
        obvious next choice), and silently embedding row 0 for it would be exactly the
        kind of quiet drift compaction must never introduce. Pinned so the strictness is
        a decision rather than an accident.
        """
        enc = _encoder(VOCAB)
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc._to_compact(t.tensor([[10, -1]], dtype=t.int64), t.tensor([[True, False]]))

    def test_OUT_OF_RANGE_id_in_a_PADDING_slot_ALSO_raises(self):
        enc = _encoder(VOCAB)
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc._to_compact(t.tensor([[10, N_SAE]], dtype=t.int64), t.tensor([[True, False]]))

    def test_an_in_range_OUT_OF_VOCAB_id_in_a_PADDING_slot_is_waved_through(self):
        """...but an in-range id that merely is not in the vocab IS fine in padding.
        This is the case the mask-awareness exists for; it must not be swept up."""
        enc = _encoder(VOCAB)
        assert 5 not in ALLOWED
        got = enc._to_compact(t.tensor([[10, 5]], dtype=t.int64), t.tensor([[True, False]]))
        assert got.tolist() == [[ALLOWED.index(10), 0]]

    def test_OOV_in_the_LAST_slot_of_the_LAST_row_still_raises(self):
        """``ok.all()`` reduces over everything. A reduction that lost an axis (``.all(0)``,
        ``.any()``) would still catch an OOV in row 0 slot 0 — the case every other test
        uses — and miss this one."""
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch(B=2, S=2, F=5, real_ids=(10, 20, 30))
        fids[-1, -1, 2] = 5  # last batch row, last segment, a mask=True slot
        assert mask[-1, -1, 2]
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc(fids, mags, mask)

    def test_a_SINGLE_real_feature_F_equals_1(self):
        """``build_no_feature_inputs`` shapes; F=1 is a real production shape and the one
        where a stray ``squeeze()`` collapses an axis."""
        enc = _encoder(VOCAB)
        out = enc(
            t.tensor([[[30]]], dtype=t.int64),
            t.zeros(1, 1, 1, dtype=t.int64),
            t.ones(1, 1, 1, dtype=t.bool),
        )
        assert out.values.shape == (1, 2 * FP8_ALIGNMENT, 32)
        assert out.layout.valid_mask[0]
        assert not out.layout.valid_mask[1:].any()

    def test_EVERY_slot_padding_with_a_NON_zero_out_of_vocab_id_is_fine(self):
        """Belt and braces on the all-masked path: not just id 0, ANY in-range id."""
        enc = _encoder(NO_ZERO)
        enc(
            t.full((1, 2, 3), 5, dtype=t.int64),  # 5 is out of NO_ZERO's vocab
            t.zeros(1, 2, 3, dtype=t.int64),
            t.zeros(1, 2, 3, dtype=t.bool),  # all padding
        )


class TestEncoderRemapAgreesWithTheVocabPrimitive:
    """Two INDEPENDENT implementations of the same map:

    * ``FeatureVocab.to_compact`` — builds ``old_to_new`` on the fly and gathers.
    * ``FeatureEncoder._to_compact`` — gathers through a cached buffer, clamps for
      memory safety, and forces padding to row 0.

    Nothing else forces them to agree, and they are used at different times (init/eval
    tooling vs. the training hot path). Cross-check them on randomized inputs.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_the_two_maps_agree_on_randomized_batches(self, seed):
        rng = random.Random(seed)
        vocab = FeatureVocab(
            n_sae_features=N_SAE,
            original_ids=tuple(sorted(rng.sample(range(N_SAE), rng.randint(1, 20)))),
        )
        enc = _encoder(vocab)

        rows, cols = rng.randint(1, 4), rng.randint(1, 8)
        allowed = list(vocab.original_ids)
        ids = t.zeros(rows, cols, dtype=t.int64)
        mask = t.zeros(rows, cols, dtype=t.bool)
        for r in range(rows):
            for c in range(cols):
                if rng.random() < 0.6:  # a REAL slot: must carry an in-vocab id
                    ids[r, c] = rng.choice(allowed)
                    mask[r, c] = True
                else:  # padding: ANY in-range id, in or out of vocab
                    ids[r, c] = rng.randrange(N_SAE)

        got = enc._to_compact(ids, mask)

        # The PUBLIC to_compact ALWAYS raises on an out-of-vocab id — there is no sentinel
        # option, because -1 is lethal under torch.compile (nn.Embedding(-1) silently
        # returns the LAST row). Padding slots here deliberately hold out-of-vocab ids, and
        # the encoder's MASK-AWARENESS is precisely what makes them safe: it knows those
        # slots are dead. The vocab primitive knows nothing of the kind, so it is fed only
        # the REAL slots — which also makes this a stronger reference than the old sentinel
        # version: it asserts the strict path accepts every id the encoder calls real.
        expected = t.zeros_like(ids)  # padding ⇒ compact row 0
        expected[mask] = vocab.to_compact(ids[mask])
        assert t.equal(got, expected)
        assert bool((got >= 0).all()), "no compact id may be negative after the remap"
        assert bool((got < vocab.size).all()), "no compact id may exceed the table"


# ===========================================================================
# SAE id 0 — the collision between "real feature" and "padding value"
# ===========================================================================


class TestSaeIdZeroCollision:
    """Every producer pads with a literal ``0``, and ``0`` is also a legitimate SAE id.
    Whether it is in the vocab changes what compact row 0 *means* — and the encoder must
    be bitwise-correct either way."""

    def test_id_0_used_as_a_REAL_feature_AND_as_padding_matches_the_reference(self):
        """The collision, at full strength: 0 is in the vocab, appears in mask=True slots,
        and simultaneously fills every padding slot. Compact row 0 therefore serves both
        roles, and the compacted model must STILL be bitwise identical to the reference.
        """
        assert 0 in ALLOWED
        full, comp = _synced_pair(VOCAB, seed=23)
        fids, mags, mask = _batch(B=2, S=2, F=6, real_ids=(0, 10, 0))
        assert (fids[~mask] == 0).all(), "padding carries id 0"
        assert (fids[mask] == 0).any(), "and so does a REAL slot"
        with t.no_grad():
            out_full = full(fids, mags, mask)
            out_comp = comp(fids, mags, mask)
        assert t.equal(out_full.layout.valid_mask, out_comp.layout.valid_mask)
        assert t.equal(out_full.layout.flat_indices, out_comp.layout.flat_indices)
        assert t.equal(out_full.values, out_comp.values)

    def test_row_0_gets_gradient_from_the_REAL_slots_only_when_id_0_is_in_vocab(self):
        """Padding must contribute exactly nothing even when it lands on a LIVE row —
        i.e. row 0's gradient must be the same whether or not padding also embeds it."""
        m = _model(VOCAB, seed=31)
        _de_degenerate(m)
        m.train()
        idx = t.randint(0, 128, (2, 16))
        tgt = t.randint(0, 128, (2, 16))
        seg_map = (t.arange(16) // 8).unsqueeze(0).expand(2, 16).contiguous()

        # F=3: exactly the real features, no padding slots at all (the FP8 pad still adds
        # masked id-0 slots, which is precisely what must contribute nothing).
        tight = _batch(B=2, S=2, F=3, real_ids=(0, 10, 20))
        m.zero_grad(set_to_none=True)
        m(idx, tgt, *tight, seg_map).backward()
        tight_grad = m.encoder.feature_embed.weight.grad.clone()  # ty: ignore[unresolved-attribute]

        # F=8: the SAME real features, now with five extra id-0 padding slots.
        padded = _batch(B=2, S=2, F=8, real_ids=(0, 10, 20))
        m.zero_grad(set_to_none=True)
        m(idx, tgt, *padded, seg_map).backward()
        padded_grad = m.encoder.feature_embed.weight.grad.clone()  # ty: ignore[unresolved-attribute]

        assert t.equal(tight_grad, padded_grad), (
            "id-0 padding slots changed row 0's gradient — padding is NOT dead"
        )
        assert tight_grad[0].abs().sum() > 0, "test is vacuous unless row 0 trains"


# ===========================================================================
# Gradients are bitwise identical, not merely 'in the right rows'
# ===========================================================================


class TestGradientBitwiseEquality:
    """``TestPaddingAndDeadRows`` proves the right ROWS receive gradient. That is a much
    weaker claim than the one #695 makes: the compacted model's gradient must be the
    reference's gradient, row-subset, BIT FOR BIT. A remap that mapped two features onto
    one row, or scaled a row, would keep the nonzero-row set intact and still be wrong.
    """

    def _grads(self, model, batch, seed=0):
        t.manual_seed(seed)
        idx = t.randint(0, 128, (2, 16))
        tgt = t.randint(0, 128, (2, 16))
        seg_map = (t.arange(16) // 8).unsqueeze(0).expand(2, 16).contiguous()
        model.train()
        model.zero_grad(set_to_none=True)
        model(idx, tgt, *batch, seg_map).backward()
        return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    def test_compact_feature_embed_GRAD_is_the_reference_grads_allowed_rows(self):
        full, comp = _synced_models(VOCAB, seed=13)
        batch = _batch(B=2, S=2, real_ids=(10, 20, 30))
        gf = self._grads(full, batch)
        gc = self._grads(comp, batch)
        rows = t.tensor(ALLOWED)
        assert t.equal(gc["encoder.feature_embed.weight"], gf["encoder.feature_embed.weight"][rows])

    def test_compact_value_embeds_GRAD_is_the_reference_grads_allowed_rows(self):
        full, comp = _synced_models(VOCAB, seed=13)
        assert len(comp.encoder.value_embeds) > 0, "vacuous unless a VE table exists"
        batch = _batch(B=2, S=2, real_ids=(10, 20, 30))
        gf = self._grads(full, batch)
        gc = self._grads(comp, batch)
        rows = t.tensor(ALLOWED)
        for k in comp.encoder.value_embeds:
            key = f"encoder.value_embeds.{k}.weight"
            assert t.equal(gc[key], gf[key][rows]), key

    def test_EVERY_other_gradient_is_bitwise_identical(self):
        full, comp = _synced_models(VOCAB, seed=13)
        batch = _batch(B=2, S=2, real_ids=(10, 20, 30))
        gf = self._grads(full, batch)
        gc = self._grads(comp, batch)
        assert gf.keys() == gc.keys()
        for name in gf:
            if "encoder.feature_embed" in name or "encoder.value_embeds" in name:
                continue
            assert t.equal(gf[name], gc[name]), f"{name} gradient diverged"

    def test_DUPLICATED_ids_accumulate_gradient_bitwise_like_the_reference(self):
        """A repeated feature id must accumulate into its one compact row exactly as the
        reference accumulates into its one full row. Index_put-style scatter bugs (last
        write wins instead of accumulate) show up here and NOWHERE else."""
        full, comp = _synced_models(VOCAB, seed=19)
        batch = _batch(B=2, S=2, F=6, real_ids=(20, 20, 20))
        gf = self._grads(full, batch)
        gc = self._grads(comp, batch)
        rows = t.tensor(ALLOWED)
        assert t.equal(gc["encoder.feature_embed.weight"], gf["encoder.feature_embed.weight"][rows])
        j = ALLOWED.index(20)
        assert gc["encoder.feature_embed.weight"][j].abs().sum() > 0, "vacuous"

    def test_the_reference_DEAD_rows_receive_exactly_zero_gradient(self):
        """The premise that licenses deleting them, stated at the encoder level."""
        full, _ = _synced_models(VOCAB, seed=13)
        gf = self._grads(full, _batch(B=2, S=2, real_ids=(10, 20, 30)))
        dead = t.tensor([i for i in range(N_SAE) if i not in set(ALLOWED)])
        g = gf["encoder.feature_embed.weight"]
        assert t.equal(g[dead], t.zeros_like(g[dead]))


# ===========================================================================
# The OOV guard, adversarially: it is the ONLY thing standing between a
# compiled model and silent corruption
# ===========================================================================


class _UnguardedEncoder(FeatureEncoder):
    """``FeatureEncoder`` with ``torch._assert_async`` REMOVED — i.e. the obvious
    implementation, the one a reviewer would wave through: the SAME searchsorted remap the
    real ``_to_compact`` computes, minus the one guard line.

    Test-only. It exists to demonstrate the corruption the guard prevents; nothing in
    production subclasses ``FeatureEncoder``.
    """

    def _to_compact(self, ids_flat, mask_flat):
        if self.feature_vocab is None:
            return ids_flat
        n = self.config.n_sae_features
        ids = self.feature_vocab_ids
        v = ids.numel()
        pos = t.searchsorted(ids, ids_flat.clamp(0, n - 1)).clamp(0, v - 1)
        hit = ids[pos] == ids_flat
        # NO torch._assert_async here — that is the point of the "unguarded" variant.
        # A non-member (hit=False) in a real slot silently resolves to compact row 0.
        return t.where(mask_flat & hit, pos, t.zeros_like(pos))


class TestTheGuardIsLoadBEARING:
    """A mutation test. Every other test here asserts the guard FIRES; these prove that
    without it a compiled model is silently, plausibly WRONG — which is the only reason
    a sync-free assert on the hot path is worth its cost.
    """

    def _unguarded(self, vocab, seed=0):
        with t.device("meta"):
            enc = _UnguardedEncoder(_enc_config(), feature_vocab=vocab)
        enc.to_empty(device="cpu")
        t.manual_seed(seed)
        enc.init_weights()
        return enc

    def test_torch_Embedding_of_minus_one_SILENTLY_returns_the_last_row_when_compiled(self):
        """The torch hazard the remap's shape is designed AROUND, pinned against torch itself.

        ``_to_compact`` deliberately emits ``where(mask & hit, pos, 0)`` — never ``-1`` — so a
        non-member resolves to compact row 0 rather than wrapping to the last row. The reason
        ``-1`` is forbidden as a sentinel is this fact: eager ``nn.Embedding(-1)`` raises, but
        compiled it wraps Python-style to the final row, silently. If a future torch made the
        compiled path raise too, this test fails and the ``-1`` prohibition (and the assert's
        cost) should be revisited. Until then it is mandatory.
        """
        dynamo.reset()
        emb = t.nn.Embedding(5, 3)
        with t.no_grad():
            emb.weight.copy_(t.arange(15, dtype=t.float32).reshape(5, 3))

        with pytest.raises(IndexError):
            emb(t.tensor([-1]))  # eager: loud

        compiled = t.compile(lambda i: emb(i), fullgraph=True)
        out = compiled(t.tensor([-1]))  # compiled: silent
        assert t.equal(out[0], emb.weight[-1]), (
            "compiled nn.Embedding(-1) no longer returns the last row — re-derive the "
            "OOV guard's justification before trusting this"
        )

    def test_WITHOUT_the_assert_a_compiled_OOV_id_becomes_COMPACT_ROW_0(self):
        """The corruption, end to end, in the encoder's own forward.

        An out-of-vocab feature does not crash and does not produce garbage — the unguarded
        remap's ``where(mask & hit, pos, 0)`` silently resolves it to COMPACT ROW 0, i.e. the
        embedding of ``original_ids[0]``, a DIFFERENT, arbitrary feature. Output is finite,
        loss is finite, training 'works'. This is what silently trains a model against the
        wrong feature dictionary.
        """
        dynamo.reset()
        enc = self._unguarded(VOCAB, seed=4)
        compiled = t.compile(enc._to_compact, fullgraph=True)

        oov = 5
        assert oov not in ALLOWED
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        bad = fids.clone()
        bad[0, 0, 0] = oov

        # The id the unguarded remap silently aliases the OOV to: compact row 0 == ALLOWED[0].
        aliased = fids.clone()
        aliased[0, 0, 0] = ALLOWED[0]

        ids_bad = bad.reshape(-1, bad.shape[-1])
        ids_aliased = aliased.reshape_as(ids_bad)
        mask_flat = mask.reshape_as(ids_bad)
        with t.no_grad():
            out_bad = compiled(ids_bad, mask_flat)
            out_aliased = compiled(ids_aliased, mask_flat)

        assert t.isfinite(out_bad).all(), "no crash, no NaN — that is the whole problem"
        assert t.equal(out_bad, out_aliased), (
            "expected the OOV id to silently become compact row 0 (original_ids[0])"
        )

    def test_and_the_REAL_encoder_raises_on_that_exact_input(self):
        """Same input, same compile settings, guard restored ⇒ loud."""
        dynamo.reset()
        enc = _encoder(VOCAB, seed=4)
        compiled = t.compile(enc._to_compact, fullgraph=True)
        fids, mags, mask = _batch(real_ids=(10, 20, 30))
        bad = fids.clone()
        bad[0, 0, 0] = 5
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            compiled(
                bad.reshape(-1, bad.shape[-1]),
                mask.reshape(-1, mask.shape[-1]),
            )


class TestCompiledGuardEdges:
    @pytest.mark.parametrize("dynamic", [False, True, None])
    def test_padding_id_0_OUTSIDE_the_vocab_does_NOT_raise_under_compile(self, dynamic):
        """The false-positive direction, on the path that actually has a trap.

        ``test_out_of_vocab_raises_UNDER_COMPILE_in_every_dynamic_mode`` proves the guard
        fires when compiled, but its vocab CONTAINS id 0 — so its padding is in-vocab and
        the mask-awareness of the guard is never exercised under compile at all. With a
        whitelist that excludes 0 (the real-world case), every padding slot carries an
        OUT-OF-VOCAB id and a mask-blind guard would reject a perfectly valid batch.
        """
        dynamo.reset()
        enc = _encoder(NO_ZERO)
        assert 0 not in NO_ZERO.original_ids
        compiled = t.compile(enc._to_compact, fullgraph=True, dynamic=dynamic)
        fids, mags, mask = _batch(F=5, real_ids=(3, 7, 10))
        assert (fids[~mask] == 0).all(), "padding carries the out-of-vocab id 0"
        compiled(
            fids.reshape(-1, fids.shape[-1]),
            mask.reshape(-1, mask.shape[-1]),
        )  # must NOT raise

    def test_compaction_adds_zero_graph_BREAKS_and_zero_extra_GRAPHS(self):
        """``torch._assert_async`` was chosen over a ``.item()`` check precisely because it
        does not break the graph.

        Strengthens the existing ``graph_break_count == 0`` check with ``graph_count == 1``:
        a guard that forced a recompile-per-batch, or a data-dependent branch that split
        the region in two, can show up as an extra GRAPH without ever being counted as a
        break.

        (Zero graph breaks *per dynamic mode* is pinned separately and more strongly by the
        ``fullgraph=True`` compiles in this class and in ``TestOutOfVocabGuard`` —
        ``fullgraph=True`` RAISES on any break, so those tests passing IS the guarantee.
        ``dynamo.explain`` cannot express a dynamic mode: its ``**kwargs`` are forwarded to
        the traced callable, not to Dynamo.)
        """
        dynamo.reset()
        enc = _encoder(VOCAB)
        fids, mags, mask = _batch()
        ids_flat = fids.reshape(-1, fids.shape[-1])
        mask_flat = mask.reshape_as(ids_flat)
        explanation = dynamo.explain(enc._to_compact)(ids_flat, mask_flat)
        assert explanation.graph_break_count == 0
        assert explanation.graph_count == 1

    def test_the_UNCOMPACTED_remap_captures_no_graph_and_no_ops(self):
        """I5: the no-vocab remap is an identity with nothing for Dynamo to capture."""
        dynamo.reset()
        enc = _encoder(None)
        fids, mags, mask = _batch()
        ids_flat = fids.reshape(-1, fids.shape[-1])
        mask_flat = mask.reshape_as(ids_flat)
        explanation = dynamo.explain(enc._to_compact)(ids_flat, mask_flat)
        assert explanation.graph_count == 0
        assert explanation.op_count == 0
        assert explanation.break_reasons == []


# ===========================================================================
# Slice-init with MULTIPLE feature-indexed tables
# ===========================================================================


class TestMultipleValueEmbeds:
    """``value_embeds`` is a SPARSE ModuleDict — ``has_ve(i, n_layer)`` decides which
    layers get one. At n_layer=2 there is exactly ONE, so every ``for k in value_embeds``
    assertion in this file is really a single-table check, and a bug in the SECOND
    feature-indexed draw (a shared closure, a stale ``s``, an RNG offset that only drifts
    on the second call) is invisible.

    n_layer=3 gives TWO tables. Pin the loop's non-vacuity while we are here.
    """

    def test_the_fixture_actually_has_value_embeds_at_all(self):
        """Guards every vacuous ``for k in comp.value_embeds`` loop in this module."""
        assert has_ve(1, 2), "n_layer=2 must produce a value_embeds table"
        assert len(_encoder(VOCAB).value_embeds) == 1

    def test_n_layer_3_produces_TWO_feature_indexed_value_embed_tables(self):
        enc = _encoder(VOCAB, n_layer=3)
        assert sorted(enc.value_embeds.keys()) == ["0", "2"]

    def test_slice_init_is_byte_exact_across_BOTH_value_embed_tables(self):
        full = _encoder(None, seed=77, n_layer=3)
        comp = _encoder(VOCAB, seed=77, n_layer=3)
        assert len(comp.value_embeds) == 2
        rows = t.tensor(ALLOWED)
        for k in comp.value_embeds:
            assert t.equal(comp.value_embeds[k].weight, full.value_embeds[k].weight[rows]), (  # ty: ignore[invalid-argument-type, not-subscriptable]
                f"value_embeds[{k}] is not the reference's allowed rows"
            )

    def test_the_rng_stream_survives_THREE_feature_indexed_draws(self):
        """feature_embed + two value_embeds. Each must restore the reference's draw count,
        and the cross-attention / decoder inits that follow depend on all three."""
        full = _encoder(None, seed=77, n_layer=3)
        comp = _encoder(VOCAB, seed=77, n_layer=3)
        fd, cd = dict(full.named_parameters()), dict(comp.named_parameters())
        assert fd.keys() == cd.keys()
        for name, full_param in fd.items():
            if "feature_embed" in name or "value_embeds" in name:
                continue
            assert t.equal(full_param, cd[name]), f"{name} diverged — RNG stream broke"

        t.manual_seed(5)
        _encoder(None, seed=5, n_layer=3)
        after_full = t.randn(8)
        t.manual_seed(5)
        _encoder(VOCAB, seed=5, n_layer=3)
        after_compact = t.randn(8)
        assert t.equal(after_full, after_compact), "RNG left at a different offset"


# ===========================================================================
# Extreme vocab shapes
# ===========================================================================


class TestExtremeVocabShapes:
    def test_a_ONE_FEATURE_vocab_builds_and_runs(self):
        """V=1: nn.Embedding(1, d). Padding and the single real feature BOTH land on row
        0, which is the only row there is."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(30,))
        enc = _encoder(vocab)
        assert enc.feature_embed.weight.shape[0] == 1
        for k in enc.value_embeds:
            assert enc.value_embeds[k].weight.shape[0] == 1  # ty: ignore[not-subscriptable]

        got = enc._to_compact(t.tensor([[30, 0]], dtype=t.int64), t.tensor([[True, False]]))
        assert got.tolist() == [[0, 0]]

        fids, mags, mask = _batch(F=4, real_ids=(30,))
        out = enc(fids, mags, mask)
        assert t.isfinite(out.values).all()

    def test_a_ONE_FEATURE_vocab_still_rejects_an_out_of_vocab_id(self):
        """The degenerate case is where a clamp-only implementation looks correct: every
        id clamps to the single valid row. The guard must still fire."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(30,))
        enc = _encoder(vocab)
        fids, mags, mask = _batch(F=4, real_ids=(30,))
        fids[0, 0, 0] = 31  # in range, NOT in vocab
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc(fids, mags, mask)

    def test_a_ONE_FEATURE_vocab_is_still_slice_init_byte_exact(self):
        full = _encoder(None, seed=61)
        comp = _encoder(FeatureVocab(n_sae_features=N_SAE, original_ids=(30,)), seed=61)
        assert t.equal(comp.feature_embed.weight, full.feature_embed.weight[30:31])

    def test_dropping_exactly_ONE_feature_is_byte_exact_too(self):
        """V = n_sae - 1, the near-identity boundary that must NOT take the None path."""
        ids = tuple(i for i in range(N_SAE) if i != 5)
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=ids)
        full, comp = _encoder(None, seed=62), _encoder(vocab, seed=62)
        assert comp.feature_embed.weight.shape[0] == N_SAE - 1
        assert t.equal(comp.feature_embed.weight, full.feature_embed.weight[t.tensor(ids)])
        assert comp._to_compact(
            t.tensor([[4, 6]], dtype=t.int64), t.ones(1, 2, dtype=t.bool)
        ).tolist() == [[4, 5]], "ids above the hole shift down by exactly one"


# ===========================================================================
# The full-table norm capture (the SAE-init rescale target)
# ===========================================================================


class TestFullNormCapture:
    def test_the_captured_norm_is_the_FULL_tables_not_the_COMPACT_tables(self):
        """A V-row subset's mean row-norm is only a sample estimate of the full table's.
        Falling back to it would make compaction silently perturb the SAE init — the one
        thing compaction must never do."""
        full, comp = _encoder(None, seed=8), _encoder(VOCAB, seed=8)
        assert t.equal(
            comp.feature_embed_full_avg_row_norm,
            _average_row_norm(full.feature_embed.weight),
        )
        compact_own = _average_row_norm(comp.feature_embed.weight)
        assert not t.equal(comp.feature_embed_full_avg_row_norm, compact_own), (
            "the fixture is toothless unless the two norms genuinely differ"
        )

    def test_the_UNCOMPACTED_encoder_captures_its_own_table(self):
        full = _encoder(None, seed=8)
        assert t.equal(
            full.feature_embed_full_avg_row_norm,
            _average_row_norm(full.feature_embed.weight),
        )

    def test_a_DOUBLE_init_weights_recaptures_the_LAST_draws_norm(self):
        """Production calls ``init_weights()`` twice. The stashed norm must belong to the
        draw that actually survives, not the discarded first one — otherwise SAE init
        rescales to a number no table ever had.

        Each model is seeded ONCE and then initialised twice, which is what a training
        process actually does. (Seeding once and interleaving the two models' second inits
        would have them draw from different offsets of the global stream — an artefact of
        the test harness, not of the code under test.)
        """
        single = _encoder(VOCAB, seed=9)
        first = single.feature_embed_full_avg_row_norm.clone()

        full, comp = _encoder_twice(None, seed=9), _encoder_twice(VOCAB, seed=9)

        assert not t.equal(comp.feature_embed_full_avg_row_norm, first), (
            "the norm was not recaptured on the second init_weights()"
        )
        assert t.equal(
            comp.feature_embed_full_avg_row_norm,
            _average_row_norm(full.feature_embed.weight),
        )
        assert t.equal(comp.feature_embed.weight, full.feature_embed.weight[t.tensor(ALLOWED)]), (
            "the second init must ALSO be slice-init byte-exact"
        )

    def test_the_SECOND_init_leaves_the_rng_stream_aligned_too(self):
        """Slice-init has to restore the draw count on EVERY call, not just the first —
        production's SAE init runs after the second one."""
        t.manual_seed(3)
        _encoder_twice(None, seed=3)
        after_full = t.randn(8)
        t.manual_seed(3)
        _encoder_twice(VOCAB, seed=3)
        after_compact = t.randn(8)
        assert t.equal(after_full, after_compact)

    def test_the_property_raises_before_init_weights_rather_than_falling_back(self):
        enc = FeatureEncoder(_enc_config(), feature_vocab=VOCAB)
        with pytest.raises(FeatureVocabError, match="never captured"):
            _ = enc.feature_embed_full_avg_row_norm


# ===========================================================================
# The FP-head × compaction gap, at the point of pain (finding 6)
# ===========================================================================


def _self_cond_config() -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(4),
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=ThresholdSelectionConfig(
            selection=ThresholdSelection(threshold_type=ThresholdType.segment_topk, threshold=2),
            precomputed_thresholds=None,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(4),
        n_sae_features=PositiveInt.parse(N_SAE),
    )


def _make_self_cond_wrapper(model) -> TargetAwareEncoderDecoderEvalWrapper:
    """Construct far enough to hit the fp-head guard. Nothing past it is reached."""
    return TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=object(),
        target=SelfConditionedTarget(),
        context_provider=object(),  # ty: ignore[invalid-argument-type]
        self_cond_config=_self_cond_config(),
    )


class TestCompactedFpHeadConsumptionSitesAreActionable:
    """Compact FP logits and inference-time self-conditioning are supported."""

    @staticmethod
    def _assert_reads_as_a_plain_missing_head(msg: str) -> None:
        assert "feature_pred_config" in msg, "must name the config knob that is missing"
        assert "compact" not in msg.lower(), (
            "an UNCOMPACTED model with no head is a different problem — a message about "
            "compaction would send the operator chasing a mask that does not exist"
        )

    def test_compacted_headless_model_reads_as_a_plain_missing_head(self):
        m = _model(VOCAB)
        assert m.feature_pred_head is None
        with pytest.raises(RuntimeError) as exc:
            m(t.zeros(1, 4, dtype=t.int64), None, return_fp_logits=True)
        self._assert_reads_as_a_plain_missing_head(str(exc.value))

    def test_return_fp_logits_on_a_compact_fp_model_returns_effective_width(self):
        m = _model(VOCAB, fp=True)
        fids, mags, mask = _batch(B=1, S=2)
        seg_map = t.tensor([[0, 0, 1, 1]], dtype=t.int64)
        logits, fp_logits = m(
            t.zeros(1, 4, dtype=t.int64),
            None,
            fids,
            mags,
            mask,
            seg_map,
            return_fp_logits=True,
        )
        assert logits.shape == (1, 4, 128)
        assert fp_logits.shape == (1, 4, VOCAB.size)

    def test_return_fp_logits_on_an_UNCOMPACTED_headless_model_reads_differently(self):
        m = _model(None)
        with pytest.raises(RuntimeError) as exc:
            m(t.zeros(1, 4, dtype=t.int64), None, return_fp_logits=True)
        self._assert_reads_as_a_plain_missing_head(str(exc.value))

    def test_eval_adapter_self_cond_accepts_a_compact_fp_model(self):
        wrapper = _make_self_cond_wrapper(
            _model(
                VOCAB,
                fp=True,
                fp_loss_mode=FeaturePredLossMode.LAST_TOKEN,
            )
        )
        assert wrapper._self_cond is not None

    def test_eval_adapter_self_cond_on_an_UNCOMPACTED_model_reads_differently(self):
        with pytest.raises(RuntimeError) as exc:
            _make_self_cond_wrapper(_model(None))
        self._assert_reads_as_a_plain_missing_head(str(exc.value))

    def test_the_UNCOMPACTED_sites_do_NOT_raise_NotImplementedError(self):
        """Teeth for the pair above: if a site raised NotImplementedError regardless of
        compaction, every 'names compaction' assertion here could be satisfied by a
        message that is simply always printed."""
        m = _model(None)
        for call in (
            lambda: m(t.zeros(1, 4, dtype=t.int64), None, return_fp_logits=True),
            lambda: _make_self_cond_wrapper(m),
        ):
            with pytest.raises(RuntimeError) as exc:
                call()
            assert not isinstance(exc.value, NotImplementedError)


# ===========================================================================
# assert_vocab_consistent: cheap, and NOT weaker for it (finding 15)
# ===========================================================================


class TestAssertVocabConsistentUsesTheResidentBuffers:
    """The guard runs after EVERY ``load_state_dict`` — every train and eval process,
    every rank. It used to rebuild the vocab maps from the Python tuple THREE times
    (``new_to_old_tensor`` directly, then ``old_to_new_tensor``, which calls
    ``new_to_old_tensor`` again internally). Measured at the production shape
    (n_sae=65,536, V=61,835): 3.53 ms, of which 3.47 ms is those rebuilds.

    The tensors it needs are already resident as the encoder's buffers. Scattering the
    RESIDENT ids buffer costs 0.06 ms.

    That is only SOUND because the ids buffer is checked against the Python vocab FIRST.
    Everything below the timing test is the proof the guard did not lose teeth.
    """

    def test_it_does_not_rebuild_the_vocab_maps_from_the_python_tuple(self, monkeypatch):
        m = _model(VOCAB)
        calls = {"new_to_old": 0, "old_to_new": 0}
        real_n2o = FeatureVocab.new_to_old_tensor
        real_o2n = FeatureVocab.old_to_new_tensor

        def counting_n2o(self, **kw):
            calls["new_to_old"] += 1
            return real_n2o(self, **kw)

        def counting_o2n(self, **kw):
            calls["old_to_new"] += 1
            return real_o2n(self, **kw)

        monkeypatch.setattr(FeatureVocab, "new_to_old_tensor", counting_n2o)
        monkeypatch.setattr(FeatureVocab, "old_to_new_tensor", counting_o2n)

        m.assert_vocab_consistent()

        assert calls["old_to_new"] == 0, (
            "the expected old->new map must be derived by SCATTERING the resident "
            "encoder.feature_vocab_ids buffer, not rebuilt from the Python tuple — "
            "old_to_new_tensor() rebuilds new_to_old a THIRD time inside itself. Got "
            f"{calls['old_to_new']} call(s)."
        )
        assert calls["new_to_old"] <= 1, (
            "the persistent ids buffer must be compared against the Python vocab exactly "
            f"once. Got {calls['new_to_old']} rebuild(s) of the id tuple."
        )

    # -- teeth: the guard must still catch every corruption it caught before --

    def test_it_still_catches_a_DIFFERENT_vocabulary_of_the_same_size(self):
        """The reason the ids-vs-configured-vocab comparison is the WHOLE guard.

        ``_to_compact`` derives its remap from ``feature_vocab_ids`` per forward, so ``ids``
        being right is the entire invariant. Overwrite the buffer with a DIFFERENT
        vocabulary of the SAME size — as ``load_state_dict(assign=True)`` from a checkpoint
        trained under a different mask would — and no shape check can save us; only the
        ids-vs-vocab comparison catches it.
        """
        m = _model(VOCAB)
        other = FeatureVocab(n_sae_features=N_SAE, original_ids=(1,) + ALLOWED[1:])
        assert other.size == VOCAB.size, "same V — no shape check can save us here"
        with t.no_grad():
            m.encoder.feature_vocab_ids.copy_(other.new_to_old_tensor())
        # The buffer is a perfectly valid, sorted, self-consistent vocabulary — it is just
        # not the one this model was configured with.
        assert m.encoder.feature_vocab_ids.tolist() == list(other.original_ids)

        with pytest.raises(FeatureVocabError, match="feature_vocab_ids"):
            m.assert_vocab_consistent()

    def test_a_clean_model_still_passes(self):
        """Anti-vacuity: the guard must not simply raise on everything."""
        _model(VOCAB).assert_vocab_consistent()
        _model(None).assert_vocab_consistent()


# ===========================================================================
# The vocab lifecycle on the PRODUCTION construction path (meta → to_empty → init)
# ===========================================================================


class TestVocabLifecycleOnTheProductionPath:
    """Pin the vocab buffers across the path production actually takes — build on ``meta``,
    ``to_empty`` onto a real device, then either ``init_weights`` (fresh) or ``init_buffers``
    + ``load_state_dict`` (resume).

    ``_to_compact`` derives its remap per-forward from the persistent ``feature_vocab_ids``
    buffer via ``searchsorted``, so ``ids`` surviving each step correctly is the WHOLE of
    correctness. Every step below is one the deleted derived-buffer machinery used to get
    wrong (D3 nested-``to_empty``, the load-then-move clobber, the V==1 P0), and is now
    structurally fine — these tests hold the ground that fix reclaimed.
    """

    IN_VOCAB = t.tensor([[10, 20, 30]], dtype=t.int64)
    IN_VOCAB_MASK = t.ones(1, 3, dtype=t.bool)
    EXPECTED = [[ALLOWED.index(i) for i in (10, 20, 30)]]

    def _meta_only(self, vocab: FeatureVocab) -> EncoderDecoderGPT:
        """Build on ``meta`` and return WITHOUT to_empty/init — the caller drives the rest."""
        dec = GPTConfig(
            sequence_len=16,
            vocab_size=128,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        )
        enc = EncoderConfig(n_embd=32, depth=2, n_head=None, n_kv_head=None, head_dim=16)
        xa = CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None)
        with t.device("meta"):
            return EncoderDecoderGPT(
                dec, enc, xa, n_sae_features=N_SAE, num_buckets=4, feature_vocab=vocab
            )

    def _assert_maps_in_vocab(self, enc: FeatureEncoder) -> None:
        assert enc._to_compact(self.IN_VOCAB, self.IN_VOCAB_MASK).tolist() == self.EXPECTED

    def _assert_oov_in_a_real_slot_raises(self, enc: FeatureEncoder, oov: int = 5) -> None:
        assert oov not in ALLOWED, "the probe id must be genuinely out of vocab"
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc._to_compact(t.tensor([[oov]], dtype=t.int64), t.ones(1, 1, dtype=t.bool))

    # 1 --------------------------------------------------------------------
    def test_1_fresh_path_maps_in_vocab_and_raises_on_oov(self):
        """meta → to_empty → seed → init_weights: the remap works, and an OOV id in a REAL
        (mask=True) slot RAISES rather than silently embedding row 0."""
        enc = _model(VOCAB, seed=0).encoder
        self._assert_maps_in_vocab(enc)
        self._assert_oov_in_a_real_slot_raises(enc)

    # 2 --------------------------------------------------------------------
    def test_2_load_path_adopts_the_checkpoint_ids_and_maps_correctly(self):
        """meta → to_empty → init_buffers → load_state_dict(strict=True, assign=True):
        ``assign=True`` REPLACES ``feature_vocab_ids`` with the checkpoint's, and the loaded
        model maps through IT."""
        ckpt = _model(VOCAB, seed=1)
        m = self._meta_only(VOCAB)
        m.to_empty(device="cpu")
        m.init_buffers()  # fresh-path buffers, OVERWRITTEN by the load below — the only order
        m.load_state_dict(ckpt.state_dict(), strict=True, assign=True)
        m.assert_vocab_consistent()
        assert t.equal(m.encoder.feature_vocab_ids, ckpt.encoder.feature_vocab_ids)
        assert m.encoder.feature_vocab_ids.tolist() == list(ALLOWED)
        self._assert_maps_in_vocab(m.encoder)

    # 3 --------------------------------------------------------------------
    def test_3_nested_parent_to_empty_then_init_is_correct(self):
        """The D3 bug site: ``parent.to_empty()`` recurses through the child's storage. Build
        the model on ``meta``, wrap it in ``_Wrapper``, materialise via the PARENT, then
        ``init_weights`` on the inner model — the remap is correct and OOV RAISES."""
        inner = self._meta_only(VOCAB)
        parent = _Wrapper(inner)
        parent.to_empty(device="cpu")  # recurses into the child, never its own to_empty
        t.manual_seed(0)
        inner.init_weights()
        inner.assert_vocab_consistent()
        self._assert_maps_in_vocab(inner.encoder)
        self._assert_oov_in_a_real_slot_raises(inner.encoder)

    # 4 --------------------------------------------------------------------
    def test_4_a_value_preserving_move_leaves_ids_and_remap_intact(self):
        """A dtype cast carries the buffer's values across, so ``feature_vocab_ids`` (int64)
        is untouched and the remap still maps correctly."""
        m = _model(VOCAB, seed=0)
        before = m.encoder.feature_vocab_ids.clone()
        m.to(dtype=t.float32)
        assert m.encoder.feature_vocab_ids.dtype == t.int64, (
            "a float cast must not touch the integer ids buffer"
        )
        assert t.equal(m.encoder.feature_vocab_ids, before)
        self._assert_maps_in_vocab(m.encoder)

    # 5 --------------------------------------------------------------------
    def test_5_deepcopy_preserves_ids_and_remap(self):
        m = _model(VOCAB, seed=0)
        m2 = copy.deepcopy(m)
        assert t.equal(m2.encoder.feature_vocab_ids, m.encoder.feature_vocab_ids)
        self._assert_maps_in_vocab(m2.encoder)

    # 6 --------------------------------------------------------------------
    def test_6_single_feature_vocab_maps_row_0_and_raises_on_oov(self):
        """V==1 (``original_ids=(0,)``) — the P0 that broke rounds 1-3. In-vocab id 0 maps to
        row 0 (the only row), and an OOV id (1) in a real slot RAISES. A clamp-only remap
        would silently accept everything here; the searchsorted membership check + assert do
        not."""
        vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=(0,))
        enc = _model(vocab, seed=0).encoder
        assert enc.feature_embed.weight.shape[0] == 1
        got = enc._to_compact(t.tensor([[0]], dtype=t.int64), t.ones(1, 1, dtype=t.bool))
        assert got.tolist() == [[0]]
        with pytest.raises(RuntimeError, match="outside the compacted vocabulary"):
            enc._to_compact(t.tensor([[1]], dtype=t.int64), t.ones(1, 1, dtype=t.bool))
