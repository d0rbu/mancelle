"""The roofline general-val LM-loss measurement RUNS on a compacted encoder (V-restricted, #720).

``penalized_lm_objective_value`` (``general_val_roofline.py``) is the forward-only
LM-loss measurement for the ROOFLINE_GCG general-val derive — the direct sibling of the
class-margin ``mean_g_steer`` (``gcg_bags.py``).  On a COMPACTED checkpoint
(``encoder.feature_vocab is not None``) the ROOFLINE_GCG derive now restricts the doc's
output-window signal bag to the checkpoint's vocabulary ``V`` (#720 PR-2 T2) BEFORE this
measurement conditions the encoder on it, so ``_to_compact`` sees only ids in ``V`` and
the forward runs correctly.  The blanket compaction guard that used to refuse this path
(#695 PR-1) is therefore LIFTED (#720 PR-2 T3).

Coverage
--------
* Compacted + V-restricted bag → the measurement RUNS and returns a finite float
  (``test_compacted_penalized_lm_objective_runs_with_in_vocab_bag``).  This is the direct
  proof the lifted guard was safe: the SAME call that raised before now completes.
* Uncompacted (byte-identity): the measurement runs and returns the same finite float it
  always did — guard removal is a strict no-op on an uncompacted checkpoint
  (``test_uncompacted_penalized_lm_objective_runs_and_returns_finite_float``).
* The encoder's own ``_to_compact`` OOV assert remains the last-line backstop: an
  out-of-``V`` bag in a real slot still trips it
  (``test_out_of_vocab_bag_still_trips_the_encoder_oov_backstop``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.common
import nanochat.flash_attention as fa_module
import nanochat.gpt
import steering.conditional.encoder_decoder.encoder as enc_module
import steering.conditional.encoder_decoder.model as model_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.general_val_roofline import (
    build_general_val_doc_member,
    penalized_lm_objective_value,
)
from steering.eval.target_conditioning import ContextFeatureSegment

_CPU = t.device("cpu")
_N_SAE = 40
_NUM_BUCKETS = 2
_SEQ_LEN = 64
_MAX_SEQ_LEN = 1024

# A genuinely COMPACTED vocab: keep only the EVEN original ids (V=20 < n_sae=40). Odd ids
# are then in-range ([0, 40)) but OUT of the vocab — the shape that trips ``_to_compact``'s
# OOV assert in a real (mask=True) slot (the retained last-line backstop).
_ALLOWED = tuple(range(0, _N_SAE, 2))
_VOCAB = FeatureVocab(n_sae_features=_N_SAE, original_ids=_ALLOWED)

# The output-window signal bag substituted at the generation slot: EVEN ids ⊆ V, so the
# COMPACTED encoder conditions on them without OOV — exactly the V-restricted bag the
# ROOFLINE_GCG derive now hands this measurement (#720 T2). Also valid on the uncompacted
# model (< n_sae=40), so both tests share it.
_BAG = FeatureConditioning.from_dict({4: 0, 12: 1})

# An OUT-of-V bag (ODD ids: in-range < n_sae=40 but ∉ V) for the encoder-backstop test: on a
# compacted encoder its real slot trips ``_to_compact``'s OOV assert.
_OOV_BAG = FeatureConditioning.from_dict({5: 0, 13: 1})

_OURS = OursGcgConfig(max_steps=8, minibatch_size=8, master_seed=0, max_forward_batch=8)


class _FakeTokenizer:
    """One-char-per-token decoder (matches the roofline unit tests' fake tokenizer)."""

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return "".join(chr(65 + int(token_id) % 26) for token_id in token_ids)


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU lowering/backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _build_model(
    monkeypatch: pytest.MonkeyPatch, *, feature_vocab: FeatureVocab | None
) -> Any:
    """A tiny CPU/float32 encdec, optionally COMPACTED via ``feature_vocab``.

    Mirrors ``test_gcg_bags_compacted_guard``'s builder so the compacted and uncompacted
    models are otherwise shape-identical and accept the SAME member + bag. ``COMPUTE_DTYPE``
    is pinned to float32 across the four modules that read it so the CPU forward is clean.
    """
    for mod in (nanochat.common, nanochat.gpt, enc_module, model_module):
        monkeypatch.setattr(mod, "COMPUTE_DTYPE", t.float32)
    dec_config = GPTConfig(
        sequence_len=_SEQ_LEN, vocab_size=100, n_layer=4, n_head=2,
        n_kv_head=2, n_embd=64, window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    t.manual_seed(1234)
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=_N_SAE, num_buckets=_NUM_BUCKETS, feature_vocab=feature_vocab,
    )
    model.init_weights()
    model.eval()
    return model


def _make_member() -> FeatureGradMember:
    """A one-doc LM-loss member under the score-time encdec layout.

    Built independently of the model (pure tensors + FeatureConditioning objects), so
    constructing it never touches the compacted encoder — the guard, not member
    construction, is what must fire first. One EMPTY-feature context segment spans the
    decoded prefix; the generation-slot placeholder bag is replaced by the ``bag`` argument
    at scoring time.
    """
    doc = _doc()
    prefix_len = len(_FakeTokenizer().decode(doc.prefix_tokens))
    return build_general_val_doc_member(
        doc,
        segments=(
            ContextFeatureSegment(
                feature_ids=(), buckets=(), char_start=0, char_end=prefix_len
            ),
        ),
        tokenizer=_FakeTokenizer(),
        output_bag=FeatureConditioning.from_dict({}),
        max_seq_len=_MAX_SEQ_LEN,
        device=_CPU,
    )


def _doc() -> Any:
    return SimpleNamespace(
        sample_id="general_val_7",
        prefix_tokens=[0, 1, 2],
        output_tokens=[3, 4],
        prefix_text="ABC",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1) Uncompacted no-op — the real forward runs and returns a finite float
# ─────────────────────────────────────────────────────────────────────────────


def test_uncompacted_penalized_lm_objective_runs_and_returns_finite_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``feature_vocab=None`` ⇒ the guard is a strict no-op and the measurement runs.

    This pins the uncompacted byte-identity requirement: Fix 1 must not change the value
    (a finite float) an uncompacted checkpoint produces.
    """
    model = _build_model(monkeypatch, feature_vocab=None)
    assert model.encoder.feature_vocab is None, "sanity: this model is NOT compacted"
    member = _make_member()

    value = penalized_lm_objective_value(
        model, member, _BAG, reg_lambda=0.25, ours=_OURS, device=_CPU
    )

    assert isinstance(value, float)
    assert value == value, "uncompacted objective must be finite (the forward really ran)"
    # It carries the feature-count penalty (len(_BAG) == 2 features at reg_lambda 0.25).
    assert value > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2) Compacted + V-restricted bag — the measurement RUNS (#720)
# ─────────────────────────────────────────────────────────────────────────────


def test_compacted_penalized_lm_objective_runs_with_in_vocab_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compacted encoder no longer refuses this measurement: with a V-restricted bag
    (``_BAG`` ⊆ V, the shape the ROOFLINE_GCG derive now hands it) it RUNS and returns a
    finite float (#720 PR-2 T3).

    Direct proof the lifted guard was safe: the SAME call that raised a
    ``NotImplementedError`` before now completes through the compacted encoder's
    ``_to_compact`` (every bag id has a compact row) and produces a real value.
    """
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    assert model.encoder.feature_vocab is not None, "sanity: this model IS compacted"
    member = _make_member()

    value = penalized_lm_objective_value(
        model, member, _BAG, reg_lambda=0.25, ours=_OURS, device=_CPU
    )

    assert isinstance(value, float)
    assert value == value, "compacted objective must be finite (the forward really ran)"
    # It carries the feature-count penalty (len(_BAG) == 2 features at reg_lambda 0.25).
    assert value > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2b) Non-empty MIXED context — the restriction runs through a REAL forward
# ─────────────────────────────────────────────────────────────────────────────
#
# ``_make_member`` (used by every test above) builds the member's context segment
# EMPTY (``feature_ids=(), buckets=()``), so ``restrict_segment_conditions_to_vocab`` is
# a structural no-op there and the encoder only ever forwards ``n_real == 0`` context
# slots. A segment-count/order/padding bug in the non-empty-context restriction path
# (``build_general_val_doc_member``'s ``restrict_segment_conditions_to_vocab`` call)
# would pass every test above. This mirrors
# ``test_gcg_bags_compacted_guard.py::test_nonempty_mixed_context_is_restricted_before_a_real_compacted_forward``
# for the general_val_roofline sibling path: it builds a member whose context segment is
# genuinely NON-EMPTY and MIXES in-V and out-of-V original ids, then runs the real
# compacted forward (``penalized_lm_objective_value``) through it.


def _mixed_context_segments_for(doc: Any) -> tuple[ContextFeatureSegment, ...]:
    """One NON-EMPTY context segment mixing IN-V (even) and OUT-of-V (odd) original
    feature ids. Even ids are in ``_ALLOWED``; odd ids are in-range (< ``_N_SAE``) but NOT
    in V -- the shape that trips ``_to_compact``'s OOV assert in a real slot if the
    restriction is bypassed. Buckets are valid for ``_NUM_BUCKETS=2`` (0 or 1)."""
    prefix_len = len(_FakeTokenizer().decode(doc.prefix_tokens))
    return (
        ContextFeatureSegment(
            feature_ids=(4, 5, 12, 13), buckets=(0, 1, 0, 1), char_start=0, char_end=prefix_len
        ),
    )


def test_nonempty_mixed_context_is_restricted_before_a_real_compacted_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member built from a NON-EMPTY, MIXED (in-V + out-of-V) context segment: the
    restriction must drop the out-of-V ids -- so ``penalized_lm_objective_value``'s real
    compacted forward does NOT hit the encoder's ``_to_compact`` OOV device-assert --
    while keeping the in-V ones, so this is a genuine ``n_real > 0`` forward, not another
    vacuous empty-context run.
    """
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    doc = _doc()
    raw_segments = _mixed_context_segments_for(doc)

    raw_ids = set(raw_segments[0].feature_ids)
    in_v_raw = raw_ids & set(_ALLOWED)
    out_of_v_raw = raw_ids - set(_ALLOWED)
    assert in_v_raw, "fixture must contain >= 1 in-V id (else the kept side is vacuous)"
    assert out_of_v_raw, "fixture must contain >= 1 out-of-V id (else the drop side is vacuous)"

    member = build_general_val_doc_member(
        doc,
        segments=raw_segments,
        tokenizer=_FakeTokenizer(),
        output_bag=_BAG,
        max_seq_len=_MAX_SEQ_LEN,
        device=_CPU,
        allowed_ids=feature_vocab_allowed_ids(model),
    )

    # (1) the restriction actually ran: the built member's context bag == the in-V subset
    #     of the raw segment -- the out-of-V ids were dropped, the in-V ones kept.
    context_bag = member.segment_conditions[0]
    restricted_ids = {fid for fid, _bucket in context_bag.entries}
    assert restricted_ids == in_v_raw
    assert restricted_ids.isdisjoint(out_of_v_raw)

    # (2) the real compacted forward (penalized_lm_objective_value) RUNS through this
    #     non-empty, restricted context and returns finite -- it does NOT hit the
    #     encoder's _to_compact OOV device-assert.
    value = penalized_lm_objective_value(
        model, member, _BAG, reg_lambda=0.25, ours=_OURS, device=_CPU
    )
    assert isinstance(value, float)
    assert value == value, (
        "objective must be finite: the forward really ran with a non-empty context"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) Encoder OOV backstop — an out-of-V bag still trips ``_to_compact`` (retained)
# ─────────────────────────────────────────────────────────────────────────────


def test_out_of_vocab_bag_still_trips_the_encoder_oov_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifted guard is not a safety regression: conditioning a compacted encoder on an
    OUT-of-V id in a real slot still fails loud at the encoder's own ``_to_compact`` OOV
    assert (encoder.py). The derive relies on V-restricting upstream (T2); this test proves
    the last-line backstop is still there if an un-restricted id ever reached the forward."""
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    assert model.encoder.feature_vocab is not None, "sanity: this model IS compacted"
    member = _make_member()

    with pytest.raises(RuntimeError):
        penalized_lm_objective_value(
            model, member, _OOV_BAG, reg_lambda=0.25, ours=_OURS, device=_CPU
        )
