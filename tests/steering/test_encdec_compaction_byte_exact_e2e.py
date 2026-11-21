"""END-TO-END byte-exactness of encoder compaction (#695).

This is the headline correctness claim, and it is proven here by RUNNING BOTH MODELS,
not by reasoning about them.

    A masked COMPACTED training run is BITWISE IDENTICAL to a masked UNCOMPACTED run:
    the same LM-loss trajectory and the same surviving weights, at EVERY step.

Two facts combine to make that true:

1. **Slice-init.** ``init_weights`` draws the FULL ``(n_sae, d)`` table — consuming the
   IDENTICAL RNG stream as the uncompacted run — then keeps only the vocab's rows. So at
   step 0 the compact table IS the row-subset of the full one, bit for bit, and every
   other parameter matches exactly.

2. **Dead rows.** Rows outside the allowed universe receive EXACTLY-zero gradient
   (cross-attention masks encoder positions as keys), so they never influence anything
   and never change. Deleting them is lossless.

Together: compaction is an EXACT IDENTITY on the trained model, not a numerical trade.

The reference model — masked but UNCOMPACTED — can no longer be produced by production
config (compaction is always-on when a mask resolves). It is constructible here because
``feature_vocab`` (a ctor arg that sizes the tables) is independent of ``feature_mask``
(a post-init attribute that gates selection).

PR-1 adds the corresponding clustered proof: its reference is the old mask-only
vocabulary, whose consolidated-away rows remain allocated, and its new model uses the
effective vocabulary, where those unreachable rows are absent. The suite compares
shared initialization, representative-only forwards (including padding-heavy and empty
segments), gradients, three real optimizer steps including AdamW state, the exact state
dict difference, SAE-init boundaries, and padding-row identity. The no-cluster model's
entire state dict remains bitwise equal to the direct mask-only constructor.

================================================================================
SCOPE OF THE BYTE-EXACTNESS CLAIM — read this before quoting it
================================================================================

Bitwise identity holds for the DEFAULT recipe (``fp8: false``,
``initialize_encoder_embeddings_from_sae: null``). It does NOT hold in two independent
cases: FP8 can observe a different padding row, while SVD deliberately derives a
different basis from the selected dictionary.

1. **FP8 (``fp8: true`` — 4xh100 / 4xh200 / 2xh200 / sweep).** NOT bitwise identical when
   SAE id 0 is outside the vocabulary. Padding carries SAE id 0, which ``_to_compact``
   maps to compact row 0 — i.e. to ``feature_embed[original_ids[0]]`` — whereas the
   uncompacted reference embeds ``feature_embed[0]``. Those are the SAME vector only when
   SAE id 0 is in the vocabulary. torchao's TENSORWISE scaling reduces over the whole
   activation — ``tensor_to_amax`` is literally ``torch.max(torch.abs(x))`` — and the
   padded feature rows are part of that tensor. A different padding row therefore yields
   a different ``amax``, hence a different quantization scale, hence different quantized
   values for every REAL activation.

   This is NOT fixable, and it is NOT a correctness defect. There is no row for SAE id 0
   in a compacted table to reproduce; forcing id 0 into the vocabulary would blunt the
   out-of-vocab guard, and zeroing padded embeddings would break the no-mask path's
   byte-identity (I5). Note also that the *uncompacted* model's padding row is itself an
   arbitrary never-trained vector, so neither run is "more correct" — the two simply round
   differently in fp8. Padding still contributes EXACTLY ZERO to the output and EXACTLY
   ZERO gradient in both.

2. **SAE-init with ``svd`` (``initialize_encoder_embeddings_from_sae: svd``).** NOT bitwise
   identical, and irreducibly so: the eigenbasis comes from the covariance of the rows you
   feed it, and requirement 6 feeds it the SUB-dictionary on purpose. No rescaling recovers
   the reference's basis. With active cluster vectors ``mean``, oriented ``W_dec`` is
   averaged before that subset, so the basis additionally describes the averaged effective
   dictionary; ``representative`` remains bitwise the pre-option SVD path.

   ``random`` — which is what the queued run actually uses — **is** bitwise identical. It
   was not originally: the rescale target used to be the *V-row* table's average row norm,
   which is only a sample estimate of the full table's (2.1e-04 relative at V=35,275 of
   65,536), so compaction silently perturbed the init. It now projects the FULL matrix,
   rescales against the FULL table's norm, and subsets LAST — running the identical GEMM
   and identical rescale as the reference. Compaction is purely representational.

Both boundaries are pinned by tests below, so they cannot drift silently.

In distributed training the exactness scope follows the production optimizer's
per-parameter collective choice: bitwise in single process always, and in DDP when the
reference/effective shapes choose the same collective. The deferred two-rank NCCL smoke
(gloo cannot drive it: DistMuonAdamW's reduce-scatter calls Work.get_future(),
unimplemented on gloo) crosses the all-reduce/reduce-scatter threshold deliberately;
two-rank reduction order remains identical. No bitwise claim is made for different
collective algorithms at three or more ranks.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as Fn
from torchao.float8 import float8_utils

import nanochat.flash_attention as _fa_mod
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.encoder import norm
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingClusterVectors,
    EncoderEmbeddingInit,
    initialize_encoder_feature_embeddings_from_matrix,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from tests.nccl_multiprocess_helpers import run_nccl_multiprocess
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

N_SAE = 256
# A realistic-shaped blacklist: keep ~22% of the dictionary, matching what the committed
# frequency band prunes on the real 65k corpus. The exact ratio is not the point; a
# realistically-SHAPED vocabulary (sparse, unsorted, id 0 sometimes absent) is.
ALLOWED = tuple(sorted({(i * 7 + 3) % N_SAE for i in range(56)}))
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
MASK = FeatureMask(
    mode=FeatureMaskMode.blacklist,
    ids=tuple(i for i in range(N_SAE) if i not in ALLOWED),
)

CLUSTER_DEFINITION = FeatureClusterDefinition(
    clusters=(
        tuple(ALLOWED[:5]),
        tuple(ALLOWED[10:14]),
        tuple(ALLOWED[20:23]),
    )
)
CLUSTER_RESOLVED = ResolvedFeatureClusters(
    source_path=Path("<byte-exact-clusters>"),
    definition=CLUSTER_DEFINITION,
    descriptor=FeatureClusterDescriptor.from_definition(
        CLUSTER_DEFINITION,
        n_sae_features=N_SAE,
    ),
)
CLUSTER_STATE = FeatureClusterConsolidation.build(CLUSTER_RESOLVED, MASK)
EFFECTIVE_VOCAB = effective_feature_vocab(CLUSTER_STATE, N_SAE)
assert EFFECTIVE_VOCAB is not None
EFFECTIVE_TO_MASK_ROWS = t.tensor(
    [VOCAB.original_ids.index(feature_id) for feature_id in EFFECTIVE_VOCAB.original_ids],
    dtype=t.int64,
)
AWAY_IDS = tuple(
    int(feature_id) for feature_id in CLUSTER_STATE.consolidated_away.nonzero().flatten().tolist()
)
AWAY_MASK_ROWS = t.tensor(
    [VOCAB.original_ids.index(feature_id) for feature_id in AWAY_IDS],
    dtype=t.int64,
)

STEPS = 8
B, T, NSEG, F = 4, 32, 4, 12
SAE_DIM = 48  # W_dec's inner dim; only the SAE-init boundary tests use it

DDP_N_SAE = 64
DDP_MASK_IDS = tuple(range(1, 36))
DDP_MASK = FeatureMask(FeatureMaskMode.whitelist, DDP_MASK_IDS)
DDP_MASK_VOCAB = FeatureVocab(DDP_N_SAE, DDP_MASK_IDS)
DDP_CLUSTER_DEFINITION = FeatureClusterDefinition(clusters=((1, 2), (3, 4), (5, 6)))
DDP_CLUSTER_RESOLVED = ResolvedFeatureClusters(
    source_path=Path("<ddp-byte-exact-clusters>"),
    definition=DDP_CLUSTER_DEFINITION,
    descriptor=FeatureClusterDescriptor.from_definition(
        DDP_CLUSTER_DEFINITION,
        n_sae_features=DDP_N_SAE,
    ),
)
DDP_CLUSTER_STATE = FeatureClusterConsolidation.build(
    DDP_CLUSTER_RESOLVED,
    DDP_MASK,
)
DDP_EFFECTIVE_VOCAB = effective_feature_vocab(DDP_CLUSTER_STATE, DDP_N_SAE)
assert DDP_EFFECTIVE_VOCAB is not None
DDP_EFFECTIVE_TO_MASK_ROWS = t.tensor(
    [
        DDP_MASK_VOCAB.original_ids.index(feature_id)
        for feature_id in DDP_EFFECTIVE_VOCAB.original_ids
    ],
    dtype=t.int64,
)


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
    old_cache_size_limit = t._dynamo.config.cache_size_limit
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    # The production trainer raises this for the model's many optimizer shape groups.
    # Real setup_optimizer steps must inherit that production precondition too.
    t._dynamo.config.cache_size_limit = 64
    yield
    _fa_mod._override_impl = old
    t._dynamo.config.cache_size_limit = old_cache_size_limit


def _sae_w_dec(seed: int = 5) -> t.Tensor:
    """An (N_SAE, SAE_DIM) decoder matrix whose row i is trivially identifiable."""
    g = t.Generator().manual_seed(seed)
    base = t.arange(N_SAE, dtype=t.float32).unsqueeze(1).expand(N_SAE, SAE_DIM).contiguous()
    return base + 0.01 * t.randn(N_SAE, SAE_DIM, generator=g)


def _build(vocab: FeatureVocab | None, seed: int) -> EncoderDecoderGPT:
    """Exactly the production path: meta device -> to_empty -> seed -> init_weights."""
    with t.device("meta"):
        m = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=T,
                vocab_size=256,
                n_layer=3,
                n_head=2,
                n_kv_head=2,
                n_embd=64,
                window_pattern="SL",
            ),
            EncoderConfig(n_embd=32, depth=3, n_head=None, n_kv_head=None, head_dim=16),
            CrossAttentionConfig(frequency=2, n_head=None, n_kv_head=None),
            n_sae_features=N_SAE,
            num_buckets=4,
            feature_vocab=vocab,
        )
    m.to_empty(device="cpu")
    t.manual_seed(seed)
    m.init_weights()
    m.feature_mask = MASK
    m.train()
    return m


def _batches(seed: int = 1234):
    """Identical data for both runs. Feature ids are drawn ONLY from the allowed
    universe (which is exactly what the masked dataloader guarantees), and the padding
    slots carry id 0 — the real-world trap."""
    g = t.Generator().manual_seed(seed)
    allowed = t.tensor(ALLOWED)
    out = []
    for _ in range(STEPS):
        idx = t.randint(0, 256, (B, T), generator=g)
        tgt = t.randint(0, 256, (B, T), generator=g)
        seg_map = (t.arange(T) // (T // NSEG)).unsqueeze(0).expand(B, T).contiguous()
        fids = t.zeros(B, NSEG, F, dtype=t.int64)
        mags = t.zeros(B, NSEG, F, dtype=t.int64)
        mask = t.zeros(B, NSEG, F, dtype=t.bool)
        for b in range(B):
            for s in range(NSEG):
                k = int(t.randint(1, F, (1,), generator=g))
                pick = allowed[t.randperm(len(allowed), generator=g)[:k]]
                fids[b, s, :k] = pick
                mags[b, s, :k] = t.randint(0, 4, (k,), generator=g)
                mask[b, s, :k] = True
        assert (fids[~mask] == 0).all(), "padding must carry id 0"
        out.append((idx, tgt, fids, mags, mask, seg_map))
    return out


def _clustered_batches(seed: int = 8128):
    """Representative-only batches, including padding-heavy and empty segments."""
    generator = t.Generator().manual_seed(seed)
    representatives = t.tensor(EFFECTIVE_VOCAB.original_ids, dtype=t.int64)  # ty: ignore[unresolved-attribute]
    batches = []
    for pattern in ("ordinary", "padding_heavy", "empty"):
        batch = 2
        n_segments = 4
        feature_slots = 8
        idx = t.randint(0, 256, (batch, T), generator=generator)
        targets = t.randint(0, 256, (batch, T), generator=generator)
        segment_map = (t.arange(T) // (T // n_segments)).unsqueeze(0).expand(batch, T).contiguous()
        feature_ids = t.zeros(batch, n_segments, feature_slots, dtype=t.int64)
        magnitudes = t.zeros(batch, n_segments, feature_slots, dtype=t.int64)
        feature_mask = t.zeros(batch, n_segments, feature_slots, dtype=t.bool)
        if pattern == "ordinary":
            for batch_index in range(batch):
                for segment_index in range(n_segments):
                    chosen = representatives[
                        t.randperm(len(representatives), generator=generator)[:4]
                    ]
                    feature_ids[batch_index, segment_index, :4] = chosen
                    magnitudes[batch_index, segment_index, :4] = t.randint(
                        0,
                        4,
                        (4,),
                        generator=generator,
                    )
                    feature_mask[batch_index, segment_index, :4] = True
        elif pattern == "padding_heavy":
            feature_ids[0, 0, 0] = representatives[0]
            magnitudes[0, 0, 0] = 2
            feature_mask[0, 0, 0] = True
        assert not any(feature_id in AWAY_IDS for feature_id in feature_ids[feature_mask])
        assert bool((feature_ids[~feature_mask] == 0).all())
        batches.append(
            (
                idx,
                targets,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
        )
    return batches


def _is_feature_indexed_parameter(name: str) -> bool:
    return name == "encoder.feature_embed.weight" or (
        name.startswith("encoder.value_embeds.") and name.endswith(".weight")
    )


def _assert_shared_tensor_equal(
    name: str,
    reference: t.Tensor,
    effective: t.Tensor,
    *,
    context: str,
) -> None:
    expected = (
        reference[EFFECTIVE_TO_MASK_ROWS] if _is_feature_indexed_parameter(name) else reference
    )
    assert t.equal(effective, expected), f"{context}: {name}"


def _assert_shared_parameters_equal(
    reference: EncoderDecoderGPT,
    effective: EncoderDecoderGPT,
    *,
    context: str,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    effective_parameters = dict(effective.named_parameters())
    assert reference_parameters.keys() == effective_parameters.keys()
    for name, reference_parameter in reference_parameters.items():
        _assert_shared_tensor_equal(
            name,
            reference_parameter,
            effective_parameters[name],
            context=context,
        )


def _assert_shared_gradients_equal(
    reference: EncoderDecoderGPT,
    effective: EncoderDecoderGPT,
    *,
    context: str,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    effective_parameters = dict(effective.named_parameters())
    for name, reference_parameter in reference_parameters.items():
        reference_grad = reference_parameter.grad
        effective_grad = effective_parameters[name].grad
        assert reference_grad is not None, f"{context}: missing reference grad for {name}"
        assert effective_grad is not None, f"{context}: missing effective grad for {name}"
        _assert_shared_tensor_equal(
            name,
            reference_grad,
            effective_grad,
            context=f"{context} gradient",
        )
        if _is_feature_indexed_parameter(name):
            assert bool((reference_grad[AWAY_MASK_ROWS] == 0).all()), (
                f"{context}: consolidated-away reference rows received gradient in {name}"
            )


def test_compacted_training_run_is_BITWISE_identical_to_the_uncompacted_reference():
    """The headline result. Run both models for real and compare every step."""
    ref = _build(None, seed=42)  # masked, UNCOMPACTED — the pre-#695 reference
    cmp_ = _build(VOCAB, seed=42)  # masked, COMPACTED   — this change

    # Step 0: slice-init must already have made them identical.
    sel = t.tensor(ALLOWED)
    ref_p, cmp_p = dict(ref.named_parameters()), dict(cmp_.named_parameters())
    assert ref_p.keys() == cmp_p.keys()
    for name, ref_param in ref_p.items():
        if "encoder.feature_embed" in name or "encoder.value_embeds" in name:
            assert t.equal(cmp_p[name], ref_param[sel]), f"step 0: {name}"
        else:
            assert t.equal(cmp_p[name], ref_param), f"step 0: {name} (RNG stream broke)"

    opt_ref = t.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.0)
    opt_cmp = t.optim.AdamW(cmp_.parameters(), lr=1e-3, weight_decay=0.0)

    losses_ref, losses_cmp = [], []
    for step, (idx, tgt, fids, mags, mask, seg_map) in enumerate(_batches()):
        lr = ref(idx, tgt, fids, mags, mask, seg_map)
        opt_ref.zero_grad(set_to_none=True)
        lr.backward()
        opt_ref.step()
        losses_ref.append(lr.item())

        lc = cmp_(idx, tgt, fids, mags, mask, seg_map)
        opt_cmp.zero_grad(set_to_none=True)
        lc.backward()
        opt_cmp.step()
        losses_cmp.append(lc.item())

        assert t.equal(lr.detach(), lc.detach()), (
            f"step {step}: LM loss diverged — {lr.item():.17g} vs {lc.item():.17g}"
        )

    assert losses_ref == losses_cmp

    # After training: the surviving weights must STILL be bitwise identical.
    ref_p, cmp_p = dict(ref.named_parameters()), dict(cmp_.named_parameters())
    for name in ref_p:
        if "encoder.feature_embed" in name or "encoder.value_embeds" in name:
            assert t.equal(cmp_p[name], ref_p[name][sel]), f"after {STEPS} steps: {name}"
        else:
            assert t.equal(cmp_p[name], ref_p[name]), f"after {STEPS} steps: {name}"


def test_the_dead_rows_never_move_which_is_WHY_deleting_them_is_lossless():
    """The uncompacted reference's DISALLOWED rows must be unchanged after training.

    If any of them moved, they would not be dead, and compaction would be dropping
    something real. This is the load-bearing premise stated as a test.
    """
    ref = _build(None, seed=42)
    before = ref.encoder.feature_embed.weight.detach().clone()
    before_ve: dict[str, t.Tensor] = {}
    for key, value_embed in ref.encoder.value_embeds.items():
        assert isinstance(value_embed, t.nn.Embedding)
        before_ve[key] = value_embed.weight.detach().clone()

    # weight_decay=0.0 — matching PRODUCTION (model.py:670 puts enc_embed_params in a
    # zero-weight-decay AdamW group). This is LOAD-BEARING for the claim below:
    # AdamW's decoupled weight decay is applied to every parameter REGARDLESS of its
    # gradient, so with the library default (0.01) the dead rows would slowly decay.
    # They would still have zero effect on outputs — compaction stays exact either way,
    # as the headline test shows — but they would not be *frozen*.
    opt = t.optim.AdamW(ref.parameters(), lr=1e-3, weight_decay=0.0)
    for idx, tgt, fids, mags, mask, seg_map in _batches():
        loss = ref(idx, tgt, fids, mags, mask, seg_map)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    dead = t.tensor([i for i in range(N_SAE) if i not in set(ALLOWED)])
    after = ref.encoder.feature_embed.weight.detach()
    assert t.equal(after[dead], before[dead]), "a disallowed feature_embed row MOVED"
    for k in ref.encoder.value_embeds:
        after_k = ref.encoder.value_embeds[k].weight.detach()  # ty: ignore[call-non-callable]
        assert t.equal(after_k[dead], before_ve[k][dead]), (
            f"a disallowed value_embeds[{k}] row MOVED"
        )

    # ...and the ALLOWED rows did move (otherwise the test proves nothing).
    live = t.tensor(list(ALLOWED))
    assert not t.equal(after[live], before[live]), "no allowed row trained — test is vacuous"


def _assert_optimizer_states_equal(
    reference_model: EncoderDecoderGPT,
    effective_model: EncoderDecoderGPT,
    reference_optimizer,
    effective_optimizer,
    *,
    context: str,
) -> None:
    reference_parameters = dict(reference_model.named_parameters())
    effective_parameters = dict(effective_model.named_parameters())
    adam_state_count = 0
    feature_adam_names: set[str] = set()
    for name, reference_parameter in reference_parameters.items():
        effective_parameter = effective_parameters[name]
        reference_state = reference_optimizer.state[reference_parameter]
        effective_state = effective_optimizer.state[effective_parameter]
        assert reference_state.keys() == effective_state.keys(), (
            f"{context}: optimizer state keys differ for {name}"
        )
        if "exp_avg" in reference_state:
            adam_state_count += 1
            assert {"exp_avg", "exp_avg_sq", "step"}.issubset(reference_state)
            if _is_feature_indexed_parameter(name):
                feature_adam_names.add(name)
        for state_key in reference_state:
            reference_value = reference_state[state_key]
            effective_value = effective_state[state_key]
            if isinstance(reference_value, t.Tensor):
                assert isinstance(effective_value, t.Tensor)
                row_indexed_state = (
                    _is_feature_indexed_parameter(name)
                    and reference_value.ndim > 0
                    and tuple(reference_value.shape) == tuple(reference_parameter.shape)
                    and tuple(effective_value.shape) == tuple(effective_parameter.shape)
                )
                expected = (
                    reference_value[EFFECTIVE_TO_MASK_ROWS]
                    if row_indexed_state
                    else reference_value
                )
                assert t.equal(effective_value, expected), (
                    f"{context}: optimizer {state_key} differs for {name}"
                )
            else:
                assert effective_value == reference_value, (
                    f"{context}: optimizer {state_key} differs for {name}"
                )
    assert adam_state_count > 0, f"{context}: no AdamW states were inspected"
    expected_feature_names = {
        name for name in reference_parameters if _is_feature_indexed_parameter(name)
    }
    assert feature_adam_names == expected_feature_names


class TestClusteredEquivalence:
    """Pre-PR-1 mask-only shape versus the effective clustered vocabulary."""

    def test_step_zero_shared_rows_and_every_other_parameter_are_bitwise(self):
        reference = _build(VOCAB, seed=2026)
        effective = _build(EFFECTIVE_VOCAB, seed=2026)
        _assert_shared_parameters_equal(reference, effective, context="step zero")

    def test_forward_logits_losses_and_gradients_are_bitwise_on_all_batch_shapes(self):
        reference = _build(VOCAB, seed=2026)
        effective = _build(EFFECTIVE_VOCAB, seed=2026)

        for batch_index, batch in enumerate(_clustered_batches()):
            idx, targets, feature_ids, magnitudes, feature_mask, segment_map = batch
            reference.zero_grad(set_to_none=True)
            effective.zero_grad(set_to_none=True)
            reference_logits = reference(
                idx,
                None,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            effective_logits = effective(
                idx,
                None,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            assert t.equal(reference_logits, effective_logits), f"batch {batch_index}: logits"
            reference_loss = reference(
                idx,
                targets,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            effective_loss = effective(
                idx,
                targets,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            assert t.equal(reference_loss, effective_loss), f"batch {batch_index}: loss"
            reference_loss.backward()
            effective_loss.backward()
            _assert_shared_gradients_equal(
                reference,
                effective,
                context=f"batch {batch_index}",
            )

    def test_three_real_production_optimizer_steps_match_weights_and_state(self):
        reference = _build(VOCAB, seed=2026)
        effective = _build(EFFECTIVE_VOCAB, seed=2026)
        reference_optimizer = reference.setup_optimizer()
        effective_optimizer = effective.setup_optimizer()

        for step, batch in enumerate(_clustered_batches()):
            idx, targets, feature_ids, magnitudes, feature_mask, segment_map = batch
            reference_optimizer.zero_grad(set_to_none=True)
            effective_optimizer.zero_grad(set_to_none=True)
            reference_loss = reference(
                idx,
                targets,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            effective_loss = effective(
                idx,
                targets,
                feature_ids,
                magnitudes,
                feature_mask,
                segment_map,
            )
            assert t.equal(reference_loss, effective_loss), f"optimizer step {step}: loss"
            reference_loss.backward()
            effective_loss.backward()
            _assert_shared_gradients_equal(
                reference,
                effective,
                context=f"optimizer step {step}",
            )
            reference_optimizer.step()
            effective_optimizer.step()
            _assert_shared_parameters_equal(
                reference,
                effective,
                context=f"optimizer step {step} post-step",
            )
            _assert_optimizer_states_equal(
                reference,
                effective,
                reference_optimizer,
                effective_optimizer,
                context=f"optimizer step {step}",
            )

    def test_state_dict_diff_is_exactly_tables_and_vocab_ids(self):
        reference = _build(VOCAB, seed=2026)
        effective = _build(EFFECTIVE_VOCAB, seed=2026)
        reference_state = reference.state_dict()
        effective_state = effective.state_dict()
        assert reference_state.keys() == effective_state.keys()
        differing = {
            key
            for key in reference_state
            if reference_state[key].shape != effective_state[key].shape
            or not t.equal(reference_state[key], effective_state[key])
        }
        expected = {
            "encoder.feature_embed.weight",
            "encoder.feature_vocab_ids",
            *{
                key
                for key in reference_state
                if key.startswith("encoder.value_embeds.") and key.endswith(".weight")
            },
        }
        assert differing == expected
        assert t.equal(
            reference_state["encoder.feature_vocab_n_sae"],
            effective_state["encoder.feature_vocab_n_sae"],
        )


class TestClusteredSaeInitBoundaries:
    def test_random_init_is_bitwise_on_every_shared_original_row(self):
        reference = _build(VOCAB, seed=3)
        effective = _build(EFFECTIVE_VOCAB, seed=3)
        matrix = _sae_w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            reference,
            matrix,
            EncoderEmbeddingInit.RANDOM,
            seed=7,
        )
        initialize_encoder_feature_embeddings_from_matrix(
            effective,
            matrix,
            EncoderEmbeddingInit.RANDOM,
            seed=7,
        )
        assert t.equal(
            effective.encoder.feature_embed.weight,
            reference.encoder.feature_embed.weight[EFFECTIVE_TO_MASK_ROWS],
        )

    def test_svd_init_intentionally_differs_between_vocabularies(self):
        reference = _build(VOCAB, seed=3)
        effective = _build(EFFECTIVE_VOCAB, seed=3)
        matrix = _sae_w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            reference,
            matrix,
            EncoderEmbeddingInit.SVD,
            seed=7,
        )
        initialize_encoder_feature_embeddings_from_matrix(
            effective,
            matrix,
            EncoderEmbeddingInit.SVD,
            seed=7,
        )
        assert not t.allclose(
            effective.encoder.feature_embed.weight,
            reference.encoder.feature_embed.weight[EFFECTIVE_TO_MASK_ROWS],
            atol=1e-3,
        )

    def test_mean_changes_only_representatives_after_full_random_rescale(self):
        representative = _build(EFFECTIVE_VOCAB, seed=3)
        mean = _build(EFFECTIVE_VOCAB, seed=3)
        matrix = _sae_w_dec()
        target_avg_norm = mean.encoder.feature_embed_full_avg_row_norm.detach().cpu()

        initialize_encoder_feature_embeddings_from_matrix(
            representative,
            matrix,
            EncoderEmbeddingInit.RANDOM,
            seed=7,
        )
        result = initialize_encoder_feature_embeddings_from_matrix(
            mean,
            matrix,
            EncoderEmbeddingInit.RANDOM,
            seed=7,
            cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
            feature_clusters=CLUSTER_STATE,
        )

        generator = t.Generator(device="cpu").manual_seed(7)
        projection = t.randn(
            SAE_DIM,
            mean.encoder.config.n_embd,
            generator=generator,
            dtype=t.float32,
        ) / (SAE_DIM**0.5)
        projected = matrix @ projection
        normalized = projected * (target_avg_norm / projected.float().norm(dim=1).mean())
        active_representatives: set[int] = set()
        expected_members = 0
        for cluster in CLUSTER_DEFINITION.clusters:
            survivors = [
                feature_id
                for feature_id in cluster
                if bool(CLUSTER_STATE.allowed_universe[feature_id])
            ]
            if not survivors:
                continue
            active_representatives.add(survivors[0])
            expected_members += len(survivors)
            row = EFFECTIVE_VOCAB.original_ids.index(survivors[0])  # ty: ignore[unresolved-attribute]
            assert t.equal(
                mean.encoder.feature_embed.weight[row],
                normalized[t.tensor(survivors)].mean(dim=0),
            )

        for row, original_id in enumerate(EFFECTIVE_VOCAB.original_ids):  # ty: ignore[unresolved-attribute]
            if original_id not in active_representatives:
                assert t.equal(
                    mean.encoder.feature_embed.weight[row],
                    representative.encoder.feature_embed.weight[row],
                )
        representative_parameters = dict(representative.named_parameters())
        mean_parameters = dict(mean.named_parameters())
        for name, representative_parameter in representative_parameters.items():
            if name != "encoder.feature_embed.weight":
                assert t.equal(mean_parameters[name], representative_parameter), name
        assert result.clusters_averaged == len(active_representatives)
        assert result.members_averaged == expected_members

    def test_svd_mean_changes_the_basis_and_only_the_feature_embedding_table(self):
        representative = _build(EFFECTIVE_VOCAB, seed=3)
        mean = _build(EFFECTIVE_VOCAB, seed=3)
        matrix = _sae_w_dec()

        initialize_encoder_feature_embeddings_from_matrix(
            representative,
            matrix,
            EncoderEmbeddingInit.SVD,
            seed=7,
            cluster_vectors=EncoderEmbeddingClusterVectors.REPRESENTATIVE,
            feature_clusters=CLUSTER_STATE,
        )
        result = initialize_encoder_feature_embeddings_from_matrix(
            mean,
            matrix,
            EncoderEmbeddingInit.SVD,
            seed=7,
            cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
            feature_clusters=CLUSTER_STATE,
        )

        representative_dictionary = matrix[t.tensor(EFFECTIVE_VOCAB.original_ids)]  # ty: ignore[unresolved-attribute]
        averaged = matrix.clone()
        expected_members = 0
        for cluster in CLUSTER_DEFINITION.clusters:
            survivors = [
                feature_id
                for feature_id in cluster
                if bool(CLUSTER_STATE.allowed_universe[feature_id])
            ]
            assert len(survivors) > 1
            averaged[survivors[0]] = matrix[t.tensor(survivors)].mean(dim=0)
            expected_members += len(survivors)
        mean_dictionary = averaged[t.tensor(EFFECTIVE_VOCAB.original_ids)]  # ty: ignore[unresolved-attribute]
        _representative_eigvals, representative_eigvecs = t.linalg.eigh(
            representative_dictionary.T @ representative_dictionary
        )
        _mean_eigvals, mean_eigvecs = t.linalg.eigh(mean_dictionary.T @ mean_dictionary)
        representative_basis = representative_eigvecs[:, -32:].flip(dims=(1,)).contiguous()
        mean_basis = mean_eigvecs[:, -32:].flip(dims=(1,)).contiguous()

        assert not t.equal(mean_basis, representative_basis)
        assert not t.equal(
            mean.encoder.feature_embed.weight,
            representative.encoder.feature_embed.weight,
        )
        representative_parameters = dict(representative.named_parameters())
        mean_parameters = dict(mean.named_parameters())
        assert representative_parameters.keys() == mean_parameters.keys()
        for name, representative_parameter in representative_parameters.items():
            if name != "encoder.feature_embed.weight":
                assert t.equal(mean_parameters[name], representative_parameter), name
        assert result.clusters_averaged == len(CLUSTER_DEFINITION.clusters)
        assert result.members_averaged == expected_members


class TestPaddingRowIdentity:
    def test_compact_row_zero_keeps_the_same_original_id_and_padding_forward(self):
        assert 0 not in VOCAB.original_ids
        assert VOCAB.original_ids[0] == EFFECTIVE_VOCAB.original_ids[0]  # ty: ignore[unresolved-attribute]
        assert int(EFFECTIVE_TO_MASK_ROWS[0]) == 0
        reference = _build(VOCAB, seed=17)
        effective = _build(EFFECTIVE_VOCAB, seed=17)
        _, _, feature_ids, magnitudes, feature_mask, _ = _clustered_batches()[1]
        with t.no_grad():
            reference_encoded = reference.encoder(
                feature_ids,
                magnitudes,
                feature_mask,
            )
            effective_encoded = effective.encoder(
                feature_ids,
                magnitudes,
                feature_mask,
            )
        assert t.equal(
            reference_encoded.layout.valid_mask,
            effective_encoded.layout.valid_mask,
        )
        assert t.equal(
            reference_encoded.layout.flat_indices,
            effective_encoded.layout.flat_indices,
        )
        assert t.equal(reference_encoded.values, effective_encoded.values)


def test_no_cluster_effective_call_builds_a_bitwise_identical_full_state_dict():
    delegated_vocab = effective_feature_vocab(MASK, N_SAE)
    assert delegated_vocab == VOCAB
    direct = _build(VOCAB, seed=991)
    delegated = _build(delegated_vocab, seed=991)
    direct_state = direct.state_dict()
    delegated_state = delegated.state_dict()
    assert direct_state.keys() == delegated_state.keys()
    for key in direct_state:
        assert t.equal(direct_state[key], delegated_state[key]), key


class TestByteExactnessBoundaries:
    """Pins EXACTLY where the bitwise claim stops holding, so it cannot drift."""

    def test_padding_embeds_a_DIFFERENT_row_when_sae_id_0_is_outside_the_vocab(self):
        """The mechanism behind the FP8 exception in the module docstring.

        Padding carries SAE id 0. `_to_compact` sends it to compact row 0, i.e. to
        `feature_embed[original_ids[0]]` — NOT to `feature_embed[0]`, which is what the
        uncompacted reference embeds. Identical only when 0 is in the vocabulary.

        Harmless for the OUTPUT (padding is masked as a key in cross-attention, so exactly
        zero contribution and exactly zero gradient — proven elsewhere in this file). It
        matters only where the padded ACTIVATION is read as a tensor-wide statistic, which
        is precisely what FP8 tensorwise scaling does.
        """
        with_zero = FeatureVocab(n_sae_features=N_SAE, original_ids=(0, 3, 7, 10))
        without_zero = FeatureVocab(n_sae_features=N_SAE, original_ids=(3, 7, 10, 20))
        assert with_zero.original_ids[0] == 0, "reference and compact agree here"
        assert without_zero.original_ids[0] != 0, "reference and compact DIVERGE here"

    def test_fp8_tensorwise_amax_spans_the_whole_activation_including_padding(self):
        """The reason a different padding row changes every REAL activation under FP8."""
        src = inspect.getsource(float8_utils.tensor_to_amax)
        assert "torch.max(torch.abs(x))" in src, (
            "torchao TENSORWISE amax is a whole-tensor reduction; if this ever becomes "
            "padding-aware, the FP8 exception in this module's docstring can be removed"
        )

    def test_sae_init_RANDOM_is_bitwise_the_reference_rows(self):
        """`random` — what the queued run uses — is INSIDE the bitwise claim.

        It was not always: the rescale target used to come from the V-row table, making a
        compacted SAE-init run a globally-rescaled projection rather than a row-subset
        (2.1e-04 relative at the real shape). It now projects the FULL matrix, rescales
        against the FULL table's norm, and subsets LAST — the identical GEMM and the
        identical rescale the reference runs. Behavioural, not source-inspection: the old
        version of this test asserted on source strings and would have passed a rewrite
        that reintroduced the bug.
        """
        ref, comp = _build(None, seed=3), _build(VOCAB, seed=3)
        w = _sae_w_dec()
        initialize_encoder_feature_embeddings_from_matrix(
            ref, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        initialize_encoder_feature_embeddings_from_matrix(
            comp, w, EncoderEmbeddingInit.RANDOM, seed=7
        )
        expected = ref.encoder.feature_embed.weight[t.tensor(VOCAB.original_ids)]
        assert t.equal(comp.encoder.feature_embed.weight, expected)

    def test_but_sae_init_SVD_is_NOT_and_cannot_be(self):
        """`svd` is the SAE-init compaction boundary, and it is irreducible: the
        eigenbasis is computed from the covariance of the rows you feed it, and the owner
        chose (correctly) to feed it the sub-dictionary. No amount of rescaling recovers
        the reference's basis. Pinned so nobody 'fixes' it by quietly widening the
        covariance back to all n_sae rows."""
        ref, comp = _build(None, seed=3), _build(VOCAB, seed=3)
        w = _sae_w_dec()
        initialize_encoder_feature_embeddings_from_matrix(ref, w, EncoderEmbeddingInit.SVD, seed=7)
        initialize_encoder_feature_embeddings_from_matrix(comp, w, EncoderEmbeddingInit.SVD, seed=7)
        expected = ref.encoder.feature_embed.weight[t.tensor(VOCAB.original_ids)]
        assert not t.allclose(comp.encoder.feature_embed.weight, expected, atol=1e-3)


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_byte_exact_under_the_PRODUCTION_recipe_bf16_cuda_and_a_DOUBLE_init():
    """The CPU/fp32/single-init E2E above does not run what production runs.

    Production (`conditional_encdec_train.py`) does:
      to_empty(cuda) -> init_weights()  [#1; ends by casting the feature tables to
                                         COMPUTE_DTYPE, a CUDA-ONLY branch]
      manual_seed(rng_seed) -> init_weights()  [#2, now over BF16 tables]

    So `_init_feature_indexed_` allocates its full (n_sae, d) scratch with
    `dtype=weight.dtype` — bf16, not fp32 — on the SECOND of two inits. The RNG stream the
    byte-exactness claim rests on is therefore a bf16 CUDA draw. Prove it there, not only
    on the CPU variant.
    """
    sel = t.tensor(ALLOWED, device="cuda")

    def build(vocab):
        with t.device("meta"):
            m = EncoderDecoderGPT(
                GPTConfig(
                    sequence_len=T,
                    vocab_size=256,
                    n_layer=3,
                    n_head=2,
                    n_kv_head=2,
                    n_embd=64,
                    window_pattern="SL",
                ),
                EncoderConfig(n_embd=32, depth=3, n_head=None, n_kv_head=None, head_dim=16),
                CrossAttentionConfig(frequency=2, n_head=None, n_kv_head=None),
                n_sae_features=N_SAE,
                num_buckets=4,
                feature_vocab=vocab,
            )
        m.to_empty(device="cuda")
        t.manual_seed(4242)
        m.init_weights()  # #1 — casts the feature tables to COMPUTE_DTYPE on CUDA
        t.manual_seed(4242)
        m.init_weights()  # #2 — slice-init now runs over the CAST tables
        return m

    ref, cmp_ = build(None), build(VOCAB)

    # The cast actually happened (otherwise this test proves nothing new).
    assert ref.encoder.feature_embed.weight.dtype != t.float32 or COMPUTE_DTYPE == t.float32

    rp, cp = dict(ref.named_parameters()), dict(cmp_.named_parameters())
    for name, reference_parameter in rp.items():
        if "encoder.feature_embed" in name or "encoder.value_embeds" in name:
            assert t.equal(cp[name], reference_parameter[sel]), f"{name} (bf16/CUDA slice-init)"
        else:
            assert t.equal(cp[name], reference_parameter), (
                f"{name} (RNG stream broke under bf16/CUDA)"
            )


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_clustered_byte_exact_under_production_bf16_cuda_double_init():
    """Clustered mask-only/effective shapes under the real CUDA initialization recipe."""
    row_map = EFFECTIVE_TO_MASK_ROWS.to("cuda")

    def build(vocab: FeatureVocab) -> EncoderDecoderGPT:
        with t.device("meta"):
            model = EncoderDecoderGPT(
                GPTConfig(
                    sequence_len=T,
                    vocab_size=256,
                    n_layer=3,
                    n_head=2,
                    n_kv_head=2,
                    n_embd=64,
                    window_pattern="SL",
                ),
                EncoderConfig(
                    n_embd=32,
                    depth=3,
                    n_head=None,
                    n_kv_head=None,
                    head_dim=16,
                ),
                CrossAttentionConfig(frequency=2, n_head=None, n_kv_head=None),
                n_sae_features=N_SAE,
                num_buckets=4,
                feature_vocab=vocab,
            )
        model.to_empty(device="cuda")
        t.manual_seed(4242)
        model.init_weights()
        t.manual_seed(4242)
        model.init_weights()
        return model

    reference = build(VOCAB)
    effective = build(EFFECTIVE_VOCAB)  # ty: ignore[invalid-argument-type]
    reference_parameters = dict(reference.named_parameters())
    effective_parameters = dict(effective.named_parameters())
    for name, reference_parameter in reference_parameters.items():
        expected = (
            reference_parameter[row_map]
            if _is_feature_indexed_parameter(name)
            else reference_parameter
        )
        assert t.equal(effective_parameters[name], expected), name


def test_production_puts_the_encoder_embeddings_in_a_ZERO_weight_decay_group():
    """The "dead rows never move" property DEPENDS on this, and it is easy to break.

    AdamW's weight decay is DECOUPLED: it is applied to every parameter regardless of
    its gradient. So with any nonzero weight decay on the encoder-embedding group, the
    disallowed rows would slowly decay toward zero even though they receive exactly-zero
    gradient. (Compaction would REMAIN exact — the headline test above holds either way,
    because the compact rows decay identically — but the rows would not be frozen, and
    the reasoning in the docs would be wrong.)

    Discovered the honest way: the first draft of the E2E test used torch's default
    AdamW (weight_decay=0.01) and the dead rows moved.
    """
    m = _build(VOCAB, seed=1)
    opt = m.setup_optimizer()
    enc_embed = set()
    for mod in (m.encoder.feature_embed, m.encoder.magnitude_embed, m.encoder.value_embeds):
        enc_embed.update(id(p) for p in mod.parameters())

    groups = [g for g in opt.param_groups if any(id(p) in enc_embed for p in g["params"])]
    assert groups, "encoder embeddings must be in some optimizer group"
    for g in groups:
        assert g.get("weight_decay", 0.0) == 0.0, (
            "encoder embeddings must have weight_decay=0.0, or the disallowed rows "
            "decay despite receiving exactly-zero gradient"
        )


def test_no_mask_run_is_byte_identical_to_the_pre_change_encoder():
    """I5. The test-owned padded reference packs exactly to the public state contract."""
    m = _build(None, seed=7)
    m.eval()
    enc = m.encoder

    def original_forward(feature_ids, magnitudes, mask):
        """encoder.forward EXACTLY as it was before #695 — no _to_compact anywhere."""
        Bx, S, Nf = feature_ids.shape
        Nfp = ((Nf + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
        pad = Nfp - Nf
        feature_ids = Fn.pad(feature_ids, (0, pad))
        magnitudes = Fn.pad(magnitudes, (0, pad))
        mask = Fn.pad(mask, (0, pad))
        ids_f = feature_ids.reshape(Bx * S, Nfp)
        mags_f = magnitudes.reshape(Bx * S, Nfp)
        mask_f = mask.reshape(Bx * S, Nfp)
        x = enc.feature_embed(ids_f) + enc.magnitude_embed(mags_f)
        for i, blk in enumerate(enc.blocks):
            ve = enc.value_embeds[str(i)](ids_f) if str(i) in enc.value_embeds else None
            x = blk(x, ve=ve, mask=mask_f)
        x = norm(x)
        return x.reshape(Bx, S, Nfp, enc.config.n_embd), mask

    for _, _, fids, mags, mask, _ in _batches(seed=99):
        with t.no_grad():
            new_out = enc(fids, mags, mask)
            old_dense, old_mask = original_forward(fids, mags, mask)
            old_out = pack_dense_encoder_output_reference(old_dense, old_mask)
        assert t.equal(new_out.layout.valid_mask, old_out.layout.valid_mask)
        assert t.equal(new_out.layout.flat_indices, old_out.layout.flat_indices)
        assert t.equal(new_out.values, old_out.values)


def _ddp_clustered_equivalence_target(rank: int, world_size: int) -> dict:
    """Two-rank production optimizer step with deliberately different collectives."""
    assert world_size == 2
    device = f"cuda:{rank}"
    _fa_mod._override_impl = FlashAttnImpl.SDPA

    def build(vocab: FeatureVocab) -> EncoderDecoderGPT:
        with t.device("meta"):
            model = EncoderDecoderGPT(
                GPTConfig(
                    sequence_len=8,
                    vocab_size=64,
                    n_layer=1,
                    n_head=2,
                    n_kv_head=2,
                    n_embd=32,
                    window_pattern="L",
                ),
                EncoderConfig(
                    n_embd=32,
                    depth=1,
                    n_head=2,
                    n_kv_head=2,
                    head_dim=16,
                ),
                CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None),
                n_sae_features=DDP_N_SAE,
                num_buckets=4,
                feature_vocab=vocab,
            )
        model.to_empty(device=device)
        t.manual_seed(73)
        if device.startswith("cuda"):
            t.cuda.manual_seed(73)
        model.init_weights()
        model.feature_mask = DDP_MASK
        model.train()
        return model

    reference = build(DDP_MASK_VOCAB)
    effective = build(DDP_EFFECTIVE_VOCAB)  # ty: ignore[invalid-argument-type]
    reference_optimizer = reference.setup_optimizer()
    effective_optimizer = effective.setup_optimizer()

    generator = t.Generator().manual_seed(900 + rank)
    idx = t.randint(0, 64, (1, 8), generator=generator).to(device)
    targets = t.randint(0, 64, (1, 8), generator=generator).to(device)
    feature_ids = t.tensor([[[1, 3, 5, 7]]], dtype=t.int64, device=device)
    magnitudes = t.tensor([[[1, 2, 3, 1]]], dtype=t.int64, device=device)
    feature_mask = t.ones(1, 1, 4, dtype=t.bool, device=device)
    segment_map = t.zeros(1, 8, dtype=t.int64, device=device)

    reference_loss = reference(
        idx,
        targets,
        feature_ids,
        magnitudes,
        feature_mask,
        segment_map,
    )
    effective_loss = effective(
        idx,
        targets,
        feature_ids,
        magnitudes,
        feature_mask,
        segment_map,
    )
    assert t.equal(reference_loss, effective_loss)
    reference_loss.backward()
    effective_loss.backward()
    reference_optimizer.step()
    effective_optimizer.step()

    reference_parameters = dict(reference.named_parameters())
    effective_parameters = dict(effective.named_parameters())
    for name, reference_parameter in reference_parameters.items():
        expected = (
            reference_parameter[DDP_EFFECTIVE_TO_MASK_ROWS]
            if _is_feature_indexed_parameter(name)
            else reference_parameter
        )
        assert t.equal(effective_parameters[name], expected), name

    return {
        "optimizer": type(effective_optimizer).__name__,
        "vocab_ids": list(DDP_EFFECTIVE_VOCAB.original_ids),  # ty: ignore[unresolved-attribute]
        "feature_embed": effective.encoder.feature_embed.weight.detach().tolist(),
    }


@pytest.mark.skipif(
    os.environ.get("RUN_DDP_SMOKE") != "1" or t.cuda.device_count() < 2,
    reason=(
        "DDP smoke needs RUN_DDP_SMOKE=1 AND >=2 CUDA devices: DistMuonAdamW's "
        "ZeRO-2 reduce-scatter calls Work.get_future(), unimplemented on gloo, "
        "so only the production NCCL backend can drive both collective paths"
    ),
)
def test_two_rank_nccl_preserves_clustered_shared_rows_across_collective_paths():
    """At two ranks, all-reduce vs reduce-scatter order is still bitwise identical."""
    assert DDP_MASK_VOCAB.size == 35
    assert DDP_EFFECTIVE_VOCAB.size == 32  # ty: ignore[unresolved-attribute]
    assert DDP_MASK_VOCAB.size % 2 == 1
    assert DDP_EFFECTIVE_VOCAB.size % 2 == 0  # ty: ignore[unresolved-attribute]
    assert DDP_EFFECTIVE_VOCAB.size * 32 >= 1024  # ty: ignore[unresolved-attribute]

    results = run_nccl_multiprocess(2, _ddp_clustered_equivalence_target)

    assert [result["optimizer"] for result in results] == [
        "DistMuonAdamW",
        "DistMuonAdamW",
    ]
    assert results[0]["vocab_ids"] == results[1]["vocab_ids"]
    assert results[0]["feature_embed"] == results[1]["feature_embed"]
