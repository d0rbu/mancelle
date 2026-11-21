"""The universal feature-GCG optimizer on a compacted checkpoint (#720 T1).

The pre-#720 optimizer derives its candidate universe from the encoder table's ROW
COUNT (``optimize._n_sae_features`` → ``feature_embed.weight.shape[0]``, fed to
``allowed_feature_ids=list(range(n_sae))``).  On a COMPACTED encoder (#695) that row
count is ``V < n_sae``, so it would propose COMPACT ids ``[0, V)`` believing them to be
ORIGINAL SAE ids — the exact silent cross-space bug the T1 change removes.

After T1 the universal optimizer:

* proposes ORIGINAL ids from ``feature_vocab_allowed_ids(model)`` (the checkpoint's ``V``),
* reads the ORIGINAL width from ``model.encoder.config.n_sae_features`` (``> |V|``),
* and scores each candidate against the ``id_weight`` row TRANSLATED from its original id
  (``old_to_new``) — proven byte-exact against the uncompacted reference in
  ``tests/steering/feature_gcg/integration/test_compacted_optimizer_equivalence.py``.

The universal optimizer and gradient path run on a compacted model directly, so
these tests reach the proposal/scoring code with no guard-bypass scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

import nanochat.flash_attention as fa_module
import steering.conditional.feature_gcg.optimize as opt_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    _materialize_allowed_embedding_pool,
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureConditioningEditSpace,
)
from steering.conditional.feature_gcg.gradient import FeatureGrad, FeatureGradMember
from steering.conditional.feature_gcg.objective import lm_loss_on_masked_tokens
from steering.conditional.feature_gcg.optimize import (
    _n_sae_features,
    optimize_feature_conditioning_universal,
)
from steering.conditional.feature_gcg.partition import PartitionBounds
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import NonEmptyNonNegativeIntTuple

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / builders
# ─────────────────────────────────────────────────────────────────────────────

N_SAE = 64
# A realistically-shaped vocabulary: sparse, unsorted at the source, V << n_sae, and
# containing compact ids (0, 3, 7, 10, 11) that DOUBLE as valid original ids — the case
# where the pre-#720 optimizer silently mis-remaps rather than crashing.
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)  # V = 12
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
NUM_BUCKETS = 4
VOCAB_SIZE = 128
SEQ_LEN = 32
T_LEN = 12
# Context (slot 0) + warm-start (gen slot) ids are drawn from V, so every conditioning
# forward is valid on the COMPACTED encoder (its ``_to_compact`` asserts in-vocab).
CONTEXT = FeatureConditioning(entries=((10, 1), (20, 2)))
WARM = FeatureConditioning(entries=((3, 1), (7, 2)))
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _build(vocab: FeatureVocab | None, *, seed: int = 0) -> EncoderDecoderGPT:
    """A tiny real ``EncoderDecoderGPT`` — meta → ``to_empty`` → seed → ``init_weights``.

    The meta-device dance is REQUIRED: ``nn.Embedding.reset_parameters`` draws
    ``numel()`` randoms at construction, so on a real device a compacted table would draw
    fewer randoms than a full one and diverge the RNG stream (which is what makes the
    slice-init byte-exactness hold — see ``test_encdec_compaction_byte_exact_e2e.py``).
    """
    dec_config = GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=2, n_head=None, n_kv_head=None, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=1, n_head=None, n_kv_head=None)
    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=N_SAE,
            num_buckets=NUM_BUCKETS,
            feature_vocab=vocab,
        )
    model.to_empty(device="cpu")
    torch.manual_seed(seed)
    model.init_weights()
    model._loaded_meta_data = {"user_config": {}}  # ty: ignore[invalid-assignment]
    return model


def _config() -> FeatureGcgConfig:
    return FeatureGcgConfig(
        max_steps=3,
        candidate_batch_size=4,
        top_k=4,
        temperature=1.0,
        master_seed=0,
        reg_lambda=0.0,
        min_k_pos=0,
        max_k_pos=6,
    )


def _member(seed: int) -> FeatureGradMember:
    """One teacher-forced completion; gen slot 1, tail 3 tokens scored."""
    gen = torch.Generator().manual_seed(seed)
    half = T_LEN // 2
    seg_map = torch.tensor([[0] * half + [1] * (T_LEN - half)], dtype=torch.long)
    dec_input = torch.randint(1, 100, (1, T_LEN), generator=gen)
    dec_target = torch.full((1, T_LEN), -1, dtype=torch.long)
    for pos in range(T_LEN - 3, T_LEN):
        dec_target[0, pos] = int(torch.randint(1, 100, (1,), generator=gen).item())
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[CONTEXT, FeatureConditioning(entries=())],
        seg_map=seg_map,
        generation_slot=1,
    )


def _members() -> list[FeatureGradMember]:
    return [_member(1), _member(2)]


def _capture_edit_space(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the optimize module's ``FeatureConditioningEditSpace`` to record the instance."""
    captured: dict[str, Any] = {}
    real_ctor = opt_module.FeatureConditioningEditSpace

    def _spy(*args: Any, **kwargs: Any) -> Any:
        space = real_ctor(*args, **kwargs)
        captured["space"] = space
        return space

    monkeypatch.setattr(opt_module, "FeatureConditioningEditSpace", _spy)
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: compacted universal optimizer proposes only ids in V and scores via V rows
# ─────────────────────────────────────────────────────────────────────────────


def test_compacted_universal_proposes_only_ids_in_V(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every accepted id is an ORIGINAL id ∈ V, the universe is ``sorted(V)``, and the
    width read is the REAL ``n_sae`` (``> |V|``) — not the compacted table height."""
    model = _build(VOCAB)
    captured = _capture_edit_space(monkeypatch)

    bag = optimize_feature_conditioning_universal(
        model,
        _members(),
        WARM,
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=DEVICE),
        config=_config(),
    )

    # (a) every accepted id is an ORIGINAL SAE id inside the vocabulary V.
    assert set(bag.feature_ids()) <= set(ALLOWED), (
        f"optimizer returned ids outside V: {set(bag.feature_ids()) - set(ALLOWED)}"
    )
    # (b) the edit space searched ORIGINAL ids sorted(V), NOT range(table_height).
    assert captured["space"].allowed_feature_ids == tuple(sorted(ALLOWED))
    assert captured["space"].allowed_feature_ids != tuple(range(len(ALLOWED)))
    # (c) the width came from encoder.config.n_sae_features (the REAL, uncompacted width).
    assert _n_sae_features(model) == N_SAE
    assert _n_sae_features(model) > len(ALLOWED)


def test_compacted_edit_space_carries_the_feature_vocab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one compacted-only branch (id_weight row translation) is armed: the edit space
    was handed the model's ``FeatureVocab`` so ``_id_weight_rows`` can translate."""
    model = _build(VOCAB)
    captured = _capture_edit_space(monkeypatch)

    optimize_feature_conditioning_universal(
        model,
        _members(),
        WARM,
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=DEVICE),
        config=_config(),
    )
    assert captured["space"].feature_vocab is model.feature_vocab
    assert captured["space"].feature_vocab is not None


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: uncompacted no-op + the mutation trap on the translation branch
# ─────────────────────────────────────────────────────────────────────────────


def test_uncompacted_universe_is_full_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identical uncompacted: the universe is ``range(n_sae)`` and no vocab is set."""
    model = _build(None)
    captured = _capture_edit_space(monkeypatch)

    optimize_feature_conditioning_universal(
        model,
        _members(),
        WARM,
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=DEVICE),
        config=_config(),
    )
    assert captured["space"].allowed_feature_ids == tuple(range(N_SAE))
    assert captured["space"].feature_vocab is None


def _edit_space(vocab: FeatureVocab | None) -> FeatureConditioningEditSpace:
    bounds = PartitionBounds(
        num_buckets=NUM_BUCKETS,
        has_negative=False,
        min_k_pos=0,
        max_k_pos=6,
        min_k_neg=0,
        max_k_neg=6,
    )
    allowed = NonEmptyNonNegativeIntTuple.parse(
        tuple(range(N_SAE)) if vocab is None else ALLOWED
    )
    return FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=allowed,
        warm_start=WARM,
        feature_vocab=vocab,
    )


def test_id_weight_rows_is_identity_when_uncompacted() -> None:
    """The literal early-return: ``_id_weight_rows(x)`` returns ``x`` UNCHANGED (same
    object) when there is no vocab, so the uncompacted gather is byte-identical."""
    space = _edit_space(None)
    x = torch.tensor([0, 3, 7, 40, 63], dtype=torch.long)
    out = space._id_weight_rows(x)
    assert out is x  # literal early-return, not a copy


def test_id_weight_rows_translates_when_compacted() -> None:
    """Mutation trap: with a vocab, ``_id_weight_rows`` does NOT return its input — it
    gathers ``old_to_new`` (original id → compact row).  Deleting the compacted branch
    (making it always identity) REDs here, never on the uncompacted test above."""
    space = _edit_space(VOCAB)
    original = torch.tensor(list(ALLOWED), dtype=torch.long)
    rows = space._id_weight_rows(original)
    # The sorted vocabulary maps to the contiguous compact rows [0, V).
    assert torch.equal(rows, torch.arange(len(ALLOWED), dtype=torch.long))
    assert not torch.equal(rows, original), "the translation branch did not fire"
    # And it matches FeatureVocab's own map exactly.
    assert torch.equal(rows, VOCAB.old_to_new_tensor()[original])


# ─────────────────────────────────────────────────────────────────────────────
# T1 review fixes (#720): fail-loud ``-1`` guard on the REMOVE id_weight gather,
# device-cached row map, and OOV-of-V precise range-check messages.
# ─────────────────────────────────────────────────────────────────────────────

# A valid SAE id (< N_SAE) that is deliberately NOT in V, so ``old_to_new`` maps it to -1.
_OOV_ID = 5
assert _OOV_ID < N_SAE and _OOV_ID not in ALLOWED


def _compacted_feature_grad(n_real: int, *, v_rows: int, seed: int = 0) -> FeatureGrad:
    """A hand-built ``FeatureGrad`` shaped like a COMPACTED encoder's: ``id_weight`` has
    ``v_rows == V`` COMPACT rows (not ``n_sae``), so a translated ``-1`` row negative-indexes
    the LAST compact row — the exact silent mis-index the REMOVE guard must prevent."""
    g = torch.Generator().manual_seed(seed)
    id_dim, d_bucket = 8, 5
    return FeatureGrad(
        id_grad=torch.randn(n_real, id_dim, generator=g),
        bucket_grad=torch.randn(n_real, d_bucket, generator=g),
        id_weight=torch.randn(v_rows, id_dim, generator=g),
        bucket_weight=torch.randn(NUM_BUCKETS, d_bucket, generator=g),
    )


def test_remove_gather_fails_loud_on_non_V_id_compacted() -> None:
    """FIX 1 (#720, IMPORTANT): the REMOVE ``id_weight`` gather RAISES — naming the offending
    ORIGINAL id — when the current bag carries a valid SAE id that is OOV of V (``old_to_new``
    → ``-1``), instead of silently indexing ``id_weight[-1]`` (the LAST compact row).

    Unreachable through the public API (``validate()`` forces the bag ⊆ V), so the non-V bag
    is built white-box and the internal REMOVE gather (``_remove_edits``) is called directly.
    MUTATION TRAP: delete the guard and ``_remove_edits`` silently returns two FeatureRemovals
    (``id_weight[-1]`` for the ``-1`` slot) — this assertion then REDs.
    """
    space = _edit_space(VOCAB)  # compacted: allowed_feature_ids = V, feature_vocab = VOCAB
    assert int(VOCAB.old_to_new_tensor()[_OOV_ID].item()) == -1
    current = FeatureConditioning(entries=((3, 1), (_OOV_ID, 2)))  # 3 ∈ V; _OOV_ID ∉ V
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    # has_negative=False, min_k_pos=0 → every slot is removable, so the -1 slot IS gathered.
    n_pos, n_neg = len(current), 0

    with pytest.raises(FeatureGcgError, match=rf"feature id {_OOV_ID}\b.*vocabulary V"):
        space._remove_edits(grad, current, n_pos, n_neg)


def test_remove_guard_is_noop_on_valid_uncompacted_bag() -> None:
    """FIX 1 is a strict no-op on an UNCOMPACTED model: a valid bag (ids < the id_weight
    height) produces one REMOVE per removable slot with NO raise — byte-identical to pre-#720
    (``_id_weight_rows`` is the identity, so no ``-1`` row can arise)."""
    space = _edit_space(None)  # uncompacted: allowed = range(N_SAE), feature_vocab = None
    current = FeatureConditioning(entries=((3, 1), (5, 2)))
    grad = _compacted_feature_grad(len(current), v_rows=N_SAE)  # full-width id_weight
    removes = space._remove_edits(grad, current, len(current), 0)
    assert {r.slot for r in removes} == {0, 1}  # ty: ignore[unresolved-attribute]


def test_add_guard_message_is_oov_of_V_on_compacted() -> None:
    """FIX 3 (#720): the ADD range-check message names the OOV-of-V ORIGINAL id (not an
    out-of-range SAE id). Reached white-box by handing ``_add_edits`` a non-V candidate."""
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=rf"feature id {_OOV_ID}\b.*vocabulary V"):
        space._add_edits(
            grad,
            current,
            top_k=4,
            magnitude=True,
            allowed_excluded=[_OOV_ID],
            n_pos=len(current),
            n_neg=0,
        )


def test_materialize_pool_message_is_oov_of_V_on_compacted() -> None:
    """FIX 3 (#720): the shared candidate-search pool guard names the OOV-of-V ORIGINAL id on
    the COMPACTED (``id_to_row``) branch — 'not in ... vocabulary V', not 'outside embedding
    vocab' (the uncompacted message, pinned verbatim elsewhere)."""
    space = _edit_space(VOCAB)
    with pytest.raises(FeatureGcgError, match=rf"original id {_OOV_ID}\b.*vocabulary V"):
        _materialize_allowed_embedding_pool(
            embedding_weight=torch.randn(len(ALLOWED), 8),  # V compact rows
            allowed_token_ids=[3, _OOV_ID],  # 3 ∈ V (row 1); _OOV_ID ∉ V (row -1)
            id_to_row=space._id_weight_rows,
        )


def test_id_weight_row_map_cached_per_device() -> None:
    """FIX 2 (#720, perf): the ``int64[n_sae]`` old_to_new map is moved to a device ONCE and
    the device-resident copy is REUSED across ``_id_weight_rows`` calls — not transferred on
    every call. Observable on CPU via the per-device cache: repeated calls resolve the SAME
    cached map object. Reverting to the per-call ``.to(device)[allowed_t]`` (no cache) REDs
    this — there is no device-resident copy to reuse."""
    space = _edit_space(VOCAB)
    ids = torch.tensor(list(ALLOWED), dtype=torch.long)

    first = space._id_weight_rows(ids)
    cached = space._id_weight_row_map_by_device[ids.device]
    second = space._id_weight_rows(ids)
    assert space._id_weight_row_map_by_device[ids.device] is cached, (
        "the device-resident row map was rebuilt/re-moved instead of reused"
    )
    # Caching must not change the translation itself.
    assert torch.equal(first, second)
    assert torch.equal(first, VOCAB.old_to_new_tensor()[ids])


# ─────────────────────────────────────────────────────────────────────────────
# T5 review (#720): FIX 1 [IMPORTANT] — ``_id_weight_rows`` had no INPUT-range guard.
# An id ``>= n_sae`` reached the bare fancy-index gather and raised a raw IndexError (on
# CUDA, plausibly a context-poisoning device assert) instead of a clean FeatureGcgError.
# A NEGATIVE id did not raise AT ALL: Python/torch fancy indexing WRAPS (``-1`` -> the
# row map's last entry) and returns a PLAUSIBLE, in-range translated row, silently
# dodging ``_require_rows_in_vocab`` — which only inspects the TRANSLATED row, never the
# input id, and runs AFTER this gather.
# ─────────────────────────────────────────────────────────────────────────────


def test_id_weight_rows_raises_clean_error_on_id_equal_to_n_sae() -> None:
    """id == n_sae is one past the row map's last valid index (shape ``(n_sae,)``).

    MUTATION TRAP: delete the guard and this raises a raw ``IndexError`` instead —
    ``pytest.raises(FeatureGcgError)`` does not catch it, so the test REDs on the wrong
    exception type.
    """
    space = _edit_space(VOCAB)
    with pytest.raises(FeatureGcgError, match=rf"feature id {N_SAE}\b") as exc:
        space._id_weight_rows(torch.tensor([N_SAE], dtype=torch.long))
    assert f"[0, {N_SAE})" in str(exc.value)


def test_id_weight_rows_accepts_the_last_valid_id() -> None:
    """Boundary control: ``n_sae - 1`` is the LAST valid id and must not raise."""
    space = _edit_space(VOCAB)
    out = space._id_weight_rows(torch.tensor([N_SAE - 1], dtype=torch.long))
    assert out.shape == (1,)


def test_id_weight_rows_raises_clean_error_on_negative_id() -> None:
    """A negative id must raise — NOT silently wrap to the row map's last entry.

    Mutation evidence: ``old_to_new[-1] == old_to_new[N_SAE - 1] == old_to_new[63] ==
    11`` — a VALID compact row (``0 <= 11 < len(ALLOWED)=12``). So without this guard,
    feeding ``-1`` to the unguarded gather does not fail ANYWHERE: it silently returns
    row 11 — as if the candidate were original id 63 — and ``_require_rows_in_vocab``
    (which only inspects the translated ROW, not the input id) sees an in-range row and
    stays quiet. MUTATION TRAP: delete the guard and this test REDs with "DID NOT RAISE"
    (no exception at all), not a wrong-message failure.
    """
    space = _edit_space(VOCAB)
    assert int(VOCAB.old_to_new_tensor()[-1].item()) == 11
    with pytest.raises(FeatureGcgError, match=r"feature id -1\b") as exc:
        space._id_weight_rows(torch.tensor([-1], dtype=torch.long))
    assert f"[0, {N_SAE})" in str(exc.value)


def test_add_edits_raises_clean_error_not_indexerror_on_id_ge_n_sae() -> None:
    """FIX 1 reached through the real ADD gather (``_add_edits`` -> ``_id_weight_rows``)."""
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=rf"feature id {N_SAE}\b"):
        space._add_edits(
            grad,
            current,
            top_k=4,
            magnitude=True,
            allowed_excluded=[N_SAE],
            n_pos=len(current),
            n_neg=0,
        )


def test_add_edits_raises_clean_error_on_negative_id() -> None:
    """FIX 1 reached through the real ADD gather with a NEGATIVE candidate id."""
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=r"feature id -1\b"):
        space._add_edits(
            grad,
            current,
            top_k=4,
            magnitude=True,
            allowed_excluded=[-1],
            n_pos=len(current),
            n_neg=0,
        )


def test_remove_edits_raises_clean_error_not_indexerror_on_id_ge_n_sae() -> None:
    """FIX 1 reached through the real REMOVE gather (``_remove_edits`` -> ``_id_weight_rows``).

    A bag CAN structurally hold ``fid == n_sae``: ``FeatureConditioning`` only checks
    ``fid >= 0``, it has no notion of ``n_sae`` — e.g. a malformed warm_start built
    white-box (as here), bypassing ``validate()``'s front-door check. (A NEGATIVE fid,
    unlike ``n_sae``, can never reach this path: ``FeatureConditioning`` itself rejects
    ``fid < 0`` at construction, so there is no REMOVE-negative case to exercise here —
    see ``test_id_weight_rows_raises_clean_error_on_negative_id`` for that guard
    directly.)
    """
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1), (N_SAE, 2)))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=rf"feature id {N_SAE}\b"):
        space._remove_edits(grad, current, len(current), 0)


def test_replace_id_raises_clean_error_not_indexerror_on_id_ge_n_sae() -> None:
    """FIX 1 reached through the real REPLACE-id gather (``_rank_replace_id`` -> the
    shared ``candidate_search`` primitive -> ``_id_weight_rows``)."""
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=rf"feature id {N_SAE}\b"):
        space._rank_replace_id(grad, current, top_k=4, allowed_excluded=[N_SAE])


def test_replace_id_raises_clean_error_on_negative_id() -> None:
    """FIX 1 reached through the real REPLACE-id gather with a NEGATIVE candidate id."""
    space = _edit_space(VOCAB)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(FeatureGcgError, match=r"feature id -1\b"):
        space._rank_replace_id(grad, current, top_k=4, allowed_excluded=[-1])


@dataclass
class _ValidateGroup:
    """Minimal ``validate()`` group: only ``key_str`` / ``members`` are read."""

    key_str: str
    members: list[Any]


def test_validate_rejects_allowed_feature_id_ge_n_sae_when_compacted() -> None:
    """FIX 1 front door: ``validate()`` rejects an ``allowed_feature_ids`` entry
    ``>= n_sae`` on a COMPACTED model BEFORE the search ever starts — independent of,
    and earlier than, the same guard inside ``_id_weight_rows``.
    """
    space = FeatureConditioningEditSpace(
        bounds=PartitionBounds(
            num_buckets=NUM_BUCKETS,
            has_negative=False,
            min_k_pos=0,
            max_k_pos=6,
            min_k_neg=0,
            max_k_neg=6,
        ),
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(
            (*ALLOWED, N_SAE)
        ),  # N_SAE is one past the valid range
        warm_start=WARM,
        feature_vocab=VOCAB,
    )
    with pytest.raises(FeatureGcgError, match=rf"feature id {N_SAE}\b") as exc:
        space.validate(_ValidateGroup(key_str="g", members=[object()]))
    assert f"[0, {N_SAE})" in str(exc.value)


def test_validate_allows_allowed_feature_ids_within_range_when_compacted() -> None:
    """Boundary control: the front-door guard must not reject valid ids in V."""
    space = _edit_space(VOCAB)  # allowed_feature_ids == ALLOWED, all < N_SAE
    validated = space.validate(_ValidateGroup(key_str="g", members=[object()]))
    assert validated.key_str == "g"


def test_validate_upper_bound_is_a_noop_when_uncompacted() -> None:
    """No ``n_sae`` is knowable from an uncompacted edit space at ``validate()`` time
    (only a real gradient's ``id_weight.shape[0]`` tells us), so the new guard must not
    fire there — byte-identical to pre-#720 (only the ``fid < 0`` check still applies).
    A very large id is accepted at ``validate()`` time; whether it is genuinely in the
    model's vocab is resolved later, at ``rank_candidates()`` time, against the real
    ``id_weight``.
    """
    space = FeatureConditioningEditSpace(
        bounds=PartitionBounds(
            num_buckets=NUM_BUCKETS,
            has_negative=False,
            min_k_pos=0,
            max_k_pos=6,
            min_k_neg=0,
            max_k_neg=6,
        ),
        allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse((3, 7, 999_999)),
        warm_start=FeatureConditioning(entries=((3, 1),)),
        feature_vocab=None,
    )
    validated = space.validate(_ValidateGroup(key_str="g", members=[object()]))
    assert validated.key_str == "g"


# ─────────────────────────────────────────────────────────────────────────────
# T5 review (#720): FIX 2 — ``_rank_replace_id`` passed ``id_to_row=self._id_weight_rows``
# UNCONDITIONALLY (never ``None``), so ``_materialize_allowed_embedding_pool``'s
# ``id_to_row is None`` discriminator — which selects the OOV error wording — always took
# the COMPACTED ("...vocabulary V...") branch, even on an UNCOMPACTED model, where the
# legacy "outside embedding vocab" wording is the correct (and previously pinned) one.
# ─────────────────────────────────────────────────────────────────────────────


def test_replace_id_oov_message_is_legacy_on_uncompacted() -> None:
    """An uncompacted edit space's REPLACE-id OOV raises the LEGACY message — never the
    'compacted model's vocabulary V' wording — even though ``_rank_replace_id`` always
    passes a non-``None`` ``id_to_row`` callable to the shared primitive.

    MUTATION TRAP: revert FIX 2 (pass ``id_to_row=self._id_weight_rows`` unconditionally)
    and this raises the COMPACTED message instead — false on an uncompacted model.
    """
    space = _edit_space(None)  # uncompacted
    v_rows = len(ALLOWED)  # arbitrary small width; only the shape matters here
    grad = _compacted_feature_grad(1, v_rows=v_rows)
    current = FeatureConditioning(entries=((3, 1),))
    oov = v_rows  # first id genuinely outside [0, v_rows)
    with pytest.raises(FeatureGcgError) as exc:
        space._rank_replace_id(grad, current, top_k=4, allowed_excluded=[oov])
    msg = str(exc.value)
    assert msg == "allowed_token_ids contain ids outside embedding vocab", (
        "must be the VERBATIM legacy message pinned in "
        "test_allowed_embedding_cache.py::test_cache_preserves_legacy_error_text_and_precedence"
    )
    assert "compacted" not in msg.lower()
    assert "vocabulary v" not in msg.lower()


def test_replace_id_oov_message_is_vocabulary_V_on_compacted() -> None:
    """The SAME REPLACE-id OOV path on a COMPACTED edit space raises the 'vocabulary V'
    message naming the original id — never the legacy wording."""
    space = _edit_space(VOCAB)  # compacted
    grad = _compacted_feature_grad(1, v_rows=len(ALLOWED))
    current = FeatureConditioning(entries=((3, 1),))
    with pytest.raises(FeatureGcgError, match=rf"original id {_OOV_ID}\b.*vocabulary V"):
        space._rank_replace_id(grad, current, top_k=4, allowed_excluded=[_OOV_ID])


# ─────────────────────────────────────────────────────────────────────────────
# Final pre-merge audit (#720 / PR #729), gap G1: ``_require_rows_in_vocab``'s
# UNCOMPACTED raise branch (``if self._id_weight_row_map is None: raise
# FeatureGcgError(uncompacted_message)``, feature_edit_space.py ~line 230-231) was never
# reached by any test above: every OOV-of-vocab test in this file uses a COMPACTED edit
# space, and the only UNCOMPACTED REMOVE test
# (``test_remove_guard_is_noop_on_valid_uncompacted_bag``) uses a VALID bag, so only the
# guard's early ``if bad_idx.numel() == 0: return`` ever fired there. On an UNCOMPACTED
# model, ``validate()``'s upper bound is a no-op (``n_sae`` is unknowable until a real
# gradient's ``id_weight`` arrives — see ``test_validate_upper_bound_is_a_noop_when_uncompacted``),
# so an id ``>= id_weight.shape[0]`` is caught HERE, at rank-candidates time, not earlier.
# ─────────────────────────────────────────────────────────────────────────────


def test_add_edits_raises_uncompacted_message_when_id_ge_id_weight_rows() -> None:
    """G1(a): the ADD path's ``_require_rows_in_vocab`` call raises the UNCOMPACTED
    message VERBATIM — never the "vocabulary V" wording (which requires a non-``None``
    ``feature_vocab``) — when a candidate id is >= ``id_weight.shape[0]`` on an
    UNCOMPACTED edit space.

    MUTATION TRAP: replace the ``raise FeatureGcgError(uncompacted_message)`` with a bare
    ``return`` (or delete the two lines) and this test REDs: execution falls through to
    ``grad.id_weight.index_select(0, rows)`` a few lines later, which raises a raw
    ``IndexError`` for the out-of-range row — a WRONG EXCEPTION TYPE
    (``pytest.raises(FeatureGcgError)`` does not catch ``IndexError``), not a wrong
    message.
    """
    space = _edit_space(None)  # uncompacted: feature_vocab=None, _id_weight_row_map=None
    current = FeatureConditioning(entries=((3, 1),))
    v_rows = 20
    grad = _compacted_feature_grad(len(current), v_rows=v_rows)
    oov_id = v_rows  # one past id_weight's row count; still < N_SAE=64 (a "valid" SAE id)
    assert oov_id < N_SAE
    with pytest.raises(FeatureGcgError) as exc:
        space._add_edits(
            grad,
            current,
            top_k=4,
            magnitude=True,
            allowed_excluded=[oov_id],
            n_pos=len(current),
            n_neg=0,
        )
    assert str(exc.value) == "allowed_feature_ids contain ids outside the SAE vocab"
    assert "vocabulary V" not in str(exc.value)


def test_add_edits_raises_uncompacted_message_on_negative_id() -> None:
    """G1(a) negative-id variant. On an UNCOMPACTED edit space ``_id_weight_rows`` is a
    pure identity with NO range check at all (only the COMPACTED branch checks the input
    range — see ``_id_weight_rows``'s own docstring), so a negative candidate id reaches
    ``_require_rows_in_vocab`` unchanged and trips the OTHER half of its ``bad`` mask
    (``rows < 0``, not ``rows >= vocab_size``) — the same guard, the other direction.

    MUTATION TRAP: with the guard deleted, ``grad.id_weight.index_select(0, rows)`` is the
    downstream gather — unlike plain tensor indexing, ``index_select`` does NOT wrap
    negative indices, so this also REDs with a raw ``IndexError`` rather than silently
    computing a wrong-row score.
    """
    space = _edit_space(None)
    current = FeatureConditioning(entries=((3, 1),))
    grad = _compacted_feature_grad(len(current), v_rows=20)
    with pytest.raises(FeatureGcgError) as exc:
        space._add_edits(
            grad,
            current,
            top_k=4,
            magnitude=True,
            allowed_excluded=[-1],
            n_pos=len(current),
            n_neg=0,
        )
    assert str(exc.value) == "allowed_feature_ids contain ids outside the SAE vocab"


def test_remove_edits_raises_uncompacted_message_when_id_ge_id_weight_rows() -> None:
    """G1(b): the REMOVE path's ``_require_rows_in_vocab`` call raises the UNCOMPACTED
    message VERBATIM when the CURRENT bag holds an id >= ``id_weight.shape[0]`` on an
    UNCOMPACTED edit space. ``FeatureConditioning`` only enforces ``fid >= 0`` at
    construction (no upper bound — see ``feature_conditioning.py``), so a legacy/malformed
    bag CAN carry such an id even though ``validate()``'s upper bound is a no-op when
    uncompacted. (A negative REMOVE id is not constructible: ``FeatureConditioning`` itself
    rejects ``fid < 0`` — see ``test_id_weight_rows_raises_clean_error_on_negative_id``'s
    docstring for that already-pinned guard.)

    MUTATION TRAP: deleting the raise turns this into a raw ``IndexError`` from the
    per-slot ``grad.id_weight[row]`` gather a few lines below — ``pytest.raises(FeatureGcgError)``
    does not catch it.
    """
    space = _edit_space(None)  # uncompacted
    v_rows = 20
    oov_id = v_rows
    current = FeatureConditioning(entries=((3, 1), (oov_id, 2)))
    grad = _compacted_feature_grad(len(current), v_rows=v_rows)
    with pytest.raises(FeatureGcgError) as exc:
        space._remove_edits(grad, current, len(current), 0)
    assert str(exc.value) == "current bag feature ids contain ids outside the SAE vocab"
    assert "vocabulary V" not in str(exc.value)


# ─────────────────────────────────────────────────────────────────────────────
# Final pre-merge audit (#720 / PR #729), gap G2: the REPLACE-id CURRENT-token
# OOV-of-V guard inside ``rank_replacements_from_gradients`` (candidate_search.py
# ~line 338-344: ``current_row < 0 or current_row >= embedding_weight.shape[0]``, the
# ``current_rows is not None`` / compacted branch) was untested. Every existing compacted
# OOV test drives a bad CANDIDATE id (the ``_materialize_allowed_embedding_pool`` guard,
# e.g. ``test_materialize_pool_message_is_oov_of_V_on_compacted``), never a bad id already
# sitting in the CURRENT BAG on the REPLACE-id path.
# ─────────────────────────────────────────────────────────────────────────────


def test_replace_id_current_token_oov_of_v_raises_named_error() -> None:
    """G2: a valid SAE id that is OOV of V, sitting in the CURRENT bag (not the candidate
    pool), reached through the REAL production path
    ``FeatureConditioningEditSpace._rank_replace_id`` -> ``rank_replacements_from_gradients``.
    Distinct from BOTH the already-tested candidate-id guard (a different function,
    ``_materialize_allowed_embedding_pool``) and G1's ``_require_rows_in_vocab``
    (ADD/REMOVE only — REPLACE-id never calls it).

    MUTATION TRAP: delete the ``if current_row < 0 or ...: raise`` in
    ``rank_replacements_from_gradients`` and this test REDs with "DID NOT RAISE" — the
    ``-1`` row silently indexes ``embedding_weight[-1]`` (the LAST compact row, a
    plausible-looking score) instead of crashing, so no exception of ANY kind is raised.
    """
    space = _edit_space(VOCAB)  # compacted
    current = FeatureConditioning(entries=((3, 1), (_OOV_ID, 2)))  # 3 ∈ V; _OOV_ID ∉ V
    grad = _compacted_feature_grad(len(current), v_rows=len(ALLOWED))
    with pytest.raises(
        FeatureGcgError,
        match=rf"current token id {_OOV_ID} at position 2 maps to embedding row -1",
    ):
        space._rank_replace_id(grad, current, top_k=4, allowed_excluded=[7])


# ─────────────────────────────────────────────────────────────────────────────
# Final pre-merge audit (#720 / PR #729), gap G3: ``rank_replacements_from_gradients``'s
# ``id_to_row`` / ``allowed_embedding_cache`` mutual-exclusion guard (candidate_search.py
# ~line 296-300) was untested: no test ever passed BOTH non-``None`` in the same call.
# ─────────────────────────────────────────────────────────────────────────────


def test_id_to_row_and_allowed_embedding_cache_cannot_be_combined() -> None:
    """G3: passing both the compacted ``id_to_row`` translator AND an
    ``allowed_embedding_cache`` (uncompacted-only) in one call raises ``FeatureGcgError``
    immediately, before any tensor work.

    MUTATION TRAP: delete the ``if id_to_row is not None and allowed_embedding_cache is
    not None: raise`` guard and this test REDs with "DID NOT RAISE": ``AllowedEmbeddingCache.resolve()``
    accepts no ``id_to_row`` parameter at all, so when a cache is bound the compacted
    translation is silently DROPPED for the candidate pool while it is still applied to
    the current tokens a few lines later — an inconsistent mix of translated and
    untranslated row spaces. In this test the raw allowed ids happen to also be valid row
    indices into the (smaller) compacted table, so the call succeeds silently end-to-end
    instead of crashing — the exact silent cross-space mis-index this optimizer exists to
    eliminate, just reached a different way.
    """
    space = _edit_space(VOCAB)
    weight = torch.randn(len(ALLOWED), 8, generator=torch.Generator().manual_seed(0))
    cache = AllowedEmbeddingCache(embedding_weight=weight, allowed_token_ids=(3, 7))
    with pytest.raises(
        FeatureGcgError,
        match="id_to_row .* and allowed_embedding_cache cannot be combined",
    ):
        rank_replacements_from_gradients(
            embedding_gradient=torch.zeros(2, 8),
            embedding_weight=weight,
            editable_positions=[1],
            allowed_token_ids=(3, 7),
            current_tokens=[0, 3],
            top_k=1,
            distance_weight=0.0,
            allowed_embedding_cache=cache,
            id_to_row=space._id_weight_rows,
        )
