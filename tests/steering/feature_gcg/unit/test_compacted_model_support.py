"""Universal feature-GCG support on compacted encoder-decoder checkpoints.

Feature-GCG reasons in original SAE-ID space. A compacted encoder has ``V``
feature-table rows indexed by compact ID, so the universal optimizer restricts
proposals to the checkpoint's original IDs in ``V`` and the edit space
translates them to compact ``id_weight`` rows only while scoring.

These tests prove that universal optimization, static-weight construction, and
feature-conditioning gradients run on compacted checkpoints; returned IDs stay
original and inside ``V``; uncompacted behavior remains full-width; and malformed
universal input still fails loudly.
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg import gradient as grad_module
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGradMember,
    _static_weights,
    _StaticWeights,
    _value_embed_keys,
)
from steering.conditional.feature_gcg.objective import lm_loss_on_masked_tokens
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from steering.conditional.feature_vocab import FeatureVocab
from tests.steering.feature_gcg._reference import feature_conditioning_gradient

# ─────────────────────────────────────────────────────────────────────────────
# Model fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_SAE = 64
ALLOWED = (0, 3, 7, 10, 11, 20, 21, 30, 31, 40, 55, 63)  # V = 12
VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=ALLOWED)
NUM_BUCKETS = 4
VOCAB_SIZE = 128
SEQ_LEN = 32
T_LEN = 12
# Conditioning IDs drawn from ALLOWED, so a compacted encoder conditions on
# them while proposal and scoring translate original IDs to compact rows.
WARM_ENTRIES = ((3, 1), (7, 2))
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _build_model(vocab: FeatureVocab | None, *, seed: int = 0) -> EncoderDecoderGPT:
    """A tiny real ``EncoderDecoderGPT`` — meta → ``to_empty`` → seed → ``init_weights``.

    The meta-device dance is REQUIRED (not decoration): ``nn.Embedding.reset_parameters``
    draws ``numel()`` randoms at construction, so on a real device a compacted table would
    draw fewer randoms than a full one and diverge the RNG stream. See
    ``tests/steering/test_encoder_feature_vocab.py::TestMetaConstructionContract``.
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
    enc_config = EncoderConfig(
        n_embd=32, depth=2, n_head=None, n_kv_head=None, head_dim=16
    )
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
    # partition_bounds_from_model reads this; absent it, every optimize raises
    # FeatureGcgError before ever reaching the code under test.
    model._loaded_meta_data = {"user_config": {}}  # ty: ignore[invalid-assignment]
    return model


@pytest.fixture
def compacted_model() -> EncoderDecoderGPT:
    return _build_model(VOCAB)


@pytest.fixture
def uncompacted_model() -> EncoderDecoderGPT:
    return _build_model(None)


def _config() -> FeatureGcgConfig:
    """A 2-step search: enough to drive the real loop, cheap enough for a unit test."""
    return FeatureGcgConfig(
        max_steps=2,
        candidate_batch_size=2,
        top_k=4,
        temperature=1.0,
        master_seed=0,
        reg_lambda=0.0,
        min_k_pos=0,
        max_k_pos=6,
    )


def _member() -> FeatureGradMember:
    torch.manual_seed(5)
    tokens = torch.randint(1, 100, (1, T_LEN + 1))
    half = T_LEN // 2
    seg_map = torch.tensor([[0] * half + [1] * (T_LEN - half)], dtype=torch.long)
    conditions = [
        FeatureConditioning(entries=WARM_ENTRIES),
        FeatureConditioning(entries=()),  # warm start at the optimized slot
    ]
    n_scored = 3
    target_mask = [False] * (T_LEN - n_scored) + [True] * n_scored
    dec_target = tokens[:, 1:].long().clone()
    dec_target[0, ~torch.tensor(target_mask, dtype=torch.bool)] = -1
    return FeatureGradMember(
        dec_input=tokens[:, :-1],
        dec_target=dec_target,
        segment_conditions=conditions,
        seg_map=seg_map,
        generation_slot=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Supported entry points, as uniform callables
# ─────────────────────────────────────────────────────────────────────────────


def _call_universal(model: Any) -> Any:
    return optimize_feature_conditioning_universal(
        model,
        [_member()],
        FeatureConditioning(entries=WARM_ENTRIES),
        objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=DEVICE),
        config=_config(),
    )


def _call_static_weights(model: Any) -> Any:
    return _static_weights(model.encoder, _value_embed_keys(model))


# The universal optimizer and static-weight builder support both compacted and
# uncompacted models.
SUPPORTED_ENTRY_POINTS = [
    pytest.param(_call_universal, id="optimize_feature_conditioning_universal"),
    pytest.param(_call_static_weights, id="gradient._static_weights"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Non-vacuity: the compacted model really is compacted
# ─────────────────────────────────────────────────────────────────────────────


class TestFixtureIsHonest:
    def test_compacted_model_tables_are_V_rows_not_n_sae(
        self, compacted_model: EncoderDecoderGPT
    ) -> None:
        """The premise of the whole path: the row count no longer means ``n_sae``.

        This is EXACTLY what a naive ``range(rows)`` candidate universe would read, so the
        universal path reads ``config.n_sae_features`` for the width and derives its
        actual proposal universe from ``feature_vocab_allowed_ids`` (V), never from the rows.
        """
        rows = int(compacted_model.encoder.feature_embed.weight.shape[0])
        assert rows == len(ALLOWED) == 12
        assert rows != N_SAE, "fixture is not compacted — every test below is vacuous"
        assert compacted_model.encoder.config.n_sae_features == N_SAE

    def test_uncompacted_model_tables_are_full_width(
        self, uncompacted_model: EncoderDecoderGPT
    ) -> None:
        assert int(uncompacted_model.encoder.feature_embed.weight.shape[0]) == N_SAE
        assert uncompacted_model.encoder.feature_vocab is None

    def test_the_dangerous_compact_ids_are_silently_valid_original_ids(self) -> None:
        """Why naive row indexing would be silently wrong (and why translation is needed).

        Compact ids 0..V-1 that are ALSO in the vocab would be remapped a second time with
        NO error — a silently-wrong optimize. This is the failure the universal path makes
        impossible by translating ORIGINAL ids to compact rows (``_id_weight_rows``) instead
        of indexing the table by raw original id.
        """
        silently_valid = [cid for cid in range(len(ALLOWED)) if cid in ALLOWED]
        assert silently_valid, (
            "no compact id doubles as an in-vocab original id — this fixture would not "
            "exercise the SILENT half of the bug the translation fixes"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Universal optimization and gradients run on a compacted model
# ─────────────────────────────────────────────────────────────────────────────


class TestCompactedModelSupport:
    def test_universal_optimize_returns_a_bag_within_V(
        self, compacted_model: EncoderDecoderGPT
    ) -> None:
        """The universal optimizer runs end to end on a compacted model, bag ⊆ V.

        This is the strongest proof the lifted guard was safe: the SAME call that raised
        before now completes and every returned feature id is an ORIGINAL id in the
        checkpoint's vocabulary V — never a compact id masquerading as an original one.
        """
        bag = _call_universal(compacted_model)
        assert isinstance(bag, FeatureConditioning)
        assert bag.entries, (
            "universal optimize returned an EMPTY bag — the in-V membership loop below would "
            "then pass vacuously, proving nothing about the proposal universe"
        )
        allowed_set = set(ALLOWED)
        for fid, bucket in bag.entries:
            assert fid in allowed_set, (
                f"universal optimize proposed feature id {fid} outside the checkpoint "
                f"vocabulary V={sorted(allowed_set)} — the proposal universe is wrong"
            )
            assert 0 <= bucket < NUM_BUCKETS

    def test_static_weights_returns_V_row_id_weight(
        self, compacted_model: EncoderDecoderGPT
    ) -> None:
        """``_static_weights`` no longer refuses a compacted encoder — it builds the ``V``-row
        ``id_weight`` the universal scoring path indexes via translated (ORIGINAL→compact)
        rows. Teeth: a resurrected guard here REDs this test."""
        encoder = compacted_model.encoder
        ve_keys = _value_embed_keys(compacted_model)
        fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
        with patch.object(grad_module, "_encoder_weight_cache", fresh):
            static = _static_weights(encoder, ve_keys)
        assert static.id_weight.shape[0] == len(ALLOWED), (
            "compacted id_weight must have exactly V rows (indexed by compact id)"
        )
        assert torch.equal(
            static.id_weight[:, : encoder.feature_embed.weight.shape[1]],
            encoder.feature_embed.weight.detach(),
        )

    def test_static_weights_runs_even_with_a_planted_cache_entry(
        self, compacted_model: EncoderDecoderGPT
    ) -> None:
        """A pre-existing (structurally valid) cache entry for a compacted encoder is served,
        not refused. With the guard gone there is no compaction check to precede the cache
        lookup; the structural invariant guard (ve_keys + column count) is what still runs."""
        encoder = compacted_model.encoder
        ve_keys = _value_embed_keys(compacted_model)
        planted = _StaticWeights(
            id_weight=torch.zeros(
                len(ALLOWED),
                encoder.feature_embed.weight.shape[1]
                + sum(int(encoder.value_embeds[k].weight.shape[1]) for k in ve_keys),  # ty: ignore[not-subscriptable]
            ),
            bucket_weight=encoder.magnitude_embed.weight.detach(),
            ve_keys=tuple(ve_keys),
            version=grad_module._weight_version(encoder, ve_keys),
        )
        fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
        fresh[encoder] = planted

        with patch.object(grad_module, "_encoder_weight_cache", fresh):
            served = _static_weights(encoder, ve_keys)
        assert served is planted, "a valid cache hit must be served for a compacted encoder"

    def test_gradient_path_runs_on_compacted_with_in_vocab_conditioning(
        self, compacted_model: EncoderDecoderGPT
    ) -> None:
        """``feature_conditioning_gradient`` runs on a compacted model when conditioning ⊆ V.

        Every conditioning id here is IN the vocab (WARM_ENTRIES ⊆ ALLOWED), so the encoder's
        forward completes and the gradient comes back carrying the ``V``-row ``id_weight`` the
        edit space then indexes by translated compact rows — no guard, no raise.
        """
        member = _member()
        grad = feature_conditioning_gradient(
            compacted_model,
            dec_input=member.dec_input,
            dec_target=member.dec_target,
            segment_conditions=member.segment_conditions,
            seg_map=member.seg_map,
            generation_slot=member.generation_slot,
            objective_scalar_fn=lambda row: row.sum(),
            device=DEVICE,
        )
        assert grad.id_weight.shape[0] == len(ALLOWED), (
            "the compacted gradient must carry a V-row id_weight"
        )
        assert grad.id_grad.shape[0] == 0, "the warm-start slot is empty → no real slots"


# ─────────────────────────────────────────────────────────────────────────────
# NEGATIVE CONTROLS — an uncompacted model must NOT be blocked (byte-identity)
# ─────────────────────────────────────────────────────────────────────────────


class TestUncompactedModelIsNotBlocked:
    @pytest.mark.parametrize("entry_point", SUPPORTED_ENTRY_POINTS)
    def test_reaches_the_normal_code_path(
        self, entry_point: Any, uncompacted_model: EncoderDecoderGPT
    ) -> None:
        """Every supported entry point reaches its normal uncompacted path."""
        result = entry_point(uncompacted_model)
        assert result is not None

    def test_universal_returns_a_real_bag(
        self, uncompacted_model: EncoderDecoderGPT
    ) -> None:
        bag = _call_universal(uncompacted_model)
        assert isinstance(bag, FeatureConditioning)
        for fid, bucket in bag.entries:
            assert 0 <= fid < N_SAE
            assert 0 <= bucket < NUM_BUCKETS

    def test_static_weights_returns_full_width_id_weight(
        self, uncompacted_model: EncoderDecoderGPT
    ) -> None:
        """The id_weight matrix keeps ONE ROW PER ORIGINAL SAE FEATURE — the invariant
        that makes ``id_grad[slot] · id_weight[f]`` mean "candidate feature f"."""
        encoder = uncompacted_model.encoder
        ve_keys = _value_embed_keys(uncompacted_model)
        fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
        with patch.object(grad_module, "_encoder_weight_cache", fresh):
            static = _static_weights(encoder, ve_keys)
        assert static.id_weight.shape[0] == N_SAE
        assert torch.equal(
            static.id_weight[:, : encoder.feature_embed.weight.shape[1]],
            encoder.feature_embed.weight.detach(),
        )

    def test_gradient_path_still_works(
        self, uncompacted_model: EncoderDecoderGPT
    ) -> None:
        """The uncompacted gradient path is byte-identical (full-width id_weight)."""
        member = _member()
        grad = feature_conditioning_gradient(
            uncompacted_model,
            dec_input=member.dec_input,
            dec_target=member.dec_target,
            segment_conditions=member.segment_conditions,
            seg_map=member.seg_map,
            generation_slot=member.generation_slot,
            objective_scalar_fn=lambda row: row.sum(),
            device=DEVICE,
        )
        assert grad.id_weight.shape[0] == N_SAE
        assert grad.id_grad.shape[0] == 0, "the warm-start slot is empty → no real slots"


# ─────────────────────────────────────────────────────────────────────────────
# Malformed universal input remains fail-loud
# ─────────────────────────────────────────────────────────────────────────────


class TestMalformedUniversalInput:
    def test_model_without_an_encoder_still_reports_the_normal_error(self) -> None:
        """The universal entry must degrade, not explode: a model with no ``encoder`` at all
        is still reported by the pre-existing ``FeatureGcgError`` path
        (``partition_bounds_from_model`` cannot read ``num_buckets``) — never an
        ``AttributeError`` leaking out."""
        headless = SimpleNamespace(_loaded_meta_data={"user_config": {}})
        with pytest.raises(FeatureGcgError):
            optimize_feature_conditioning_universal(
                headless,
                [_member()],
                FeatureConditioning(entries=WARM_ENTRIES),
                objective_factory=lambda m: lm_loss_on_masked_tokens(m, device=DEVICE),
                config=_config(),
            )
