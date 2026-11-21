"""The forward-only g_steer derive RUNS on a compacted encoder (V-restricted, #720 PR-2).

Feature-DiffMean's fair/roofline bag DERIVE measures g_steer via the forward-only
``mean_g_steer``.  On a COMPACTED checkpoint (``encoder.feature_vocab is not None``) the
derive now restricts every bag/context it conditions on to the checkpoint's vocabulary
``V`` BEFORE that forward (warm bags via ``feature_diffmean_bag(allowed_ids=...)``, #720 T2;
optimized bags via the V-restricted universal optimizer, #720 T1), so the encoder's
``_to_compact`` sees only ids in ``V`` — no OOV — and the derive completes producing bags
whose feature ids lie in ``V``.  The blanket compaction guard that used to refuse
``mean_g_steer`` (#695 PR-1) is therefore LIFTED (#720 PR-2 T3).

Choke point + coverage
----------------------
``mean_g_steer`` (``gcg_bags.py``) is the single forward-only measurement both derives
reach, via ONE ``group_g_steer = _group_g_steer(model, ours, mode, device)`` —
a ``functools.partial(mean_g_steer, model, ...)``.  Coverage:

* ``mean_g_steer`` and the shared ``_group_g_steer`` partial RUN on a compacted model with a
  V-restricted bag (``test_compacted_mean_g_steer_runs_with_in_vocab_bag``,
  ``test_group_g_steer_partial_runs_on_compacted_encoder``).
* Each named derive completes end-to-end and writes bags whose ids ⊆ ``V``
  (``test_{fair,roofline}_derive_runs_on_compacted_encoder``), threading
  ``allowed_ids=feature_vocab_allowed_ids(model)`` into the warm ``feature_diffmean_bag``
  call (``test_{fair,roofline}_derive_threads_allowed_ids_into_warm_diffmean_call``).
* The encoder's own ``_to_compact`` OOV assert remains the last-line backstop: an
  out-of-``V`` bag in a real slot still trips it
  (``test_out_of_vocab_bag_still_trips_the_encoder_oov_backstop``).
* Uncompacted is a strict no-op — the real forward runs unchanged
  (``test_uncompacted_mean_g_steer_runs_and_returns_finite_g_steer``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch as t

import nanochat.common
import nanochat.flash_attention as fa_module
import nanochat.gpt
import steering.conditional.encoder_decoder.encoder as enc_module
import steering.conditional.encoder_decoder.model as model_module
import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.tokenization import encode_prefix
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.gcg_bags import (
    build_ours_gcg_bags_fair,
    build_ours_gcg_bags_roofline,
    mean_g_steer,
)
from steering.eval.axbench_bpb_adv.gcg_members import build_class_margin_members
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_bpb_adv.tokenizer_stubs import StubBaseModel, StubNanochatTokenizer
from tests.runtime_attribute_helpers import set_runtime_attribute

_CPU = t.device("cpu")
_N_SAE = 40
_NUM_BUCKETS = 2
_SEQ_LEN = 64
_MAX_SEQ_LEN = 1024

# A genuinely COMPACTED vocab: keep only the EVEN original ids (V=20 < n_sae=40).
# Odd ids are then in-range ([0, 40)) but OUT of the vocab — the shape that trips
# ``_to_compact``'s OOV assert in a real (mask=True) slot (the retained last-line backstop).
_ALLOWED = tuple(range(0, _N_SAE, 2))
_VOCAB = FeatureVocab(n_sae_features=_N_SAE, original_ids=_ALLOWED)

# Bag feature ids are EVEN ⊆ V, so the COMPACTED encoder conditions on them without OOV —
# the V-restricted bag shape both derives now hand ``mean_g_steer``. Also valid on the
# uncompacted model (< n_sae=40), so both tests share it.
_BAG = FeatureConditioning.from_dict({4: 0, 12: 1})

# An OUT-of-V bag (ODD ids: in-range < n_sae=40 but ∉ V) for the encoder-backstop test: on a
# compacted encoder its real slot trips ``_to_compact``'s OOV assert.
_OOV_BAG = FeatureConditioning.from_dict({5: 0, 13: 1})

# Canonical training-SAE block matching the default ``neuronpedia_source``
# "8-gemmascope-res-16k"; only the full-derive tests (which run resolve_k /
# assert_single_positive_bucket) read it.
_SAE_META = {
    "release": "gemmascope",
    "model": "gemma-2-2b",
    "activation_type": "residual",
    "layer": 8,
    "width": "16k",
    "l0": None,
}
_USER_CONFIG = {
    "feature_selection": {"distribution": "geometric", "mean": 5.0, "min_k": 1, "max_k": 25},
    "negative_selection": {"distribution": "geometric", "mean": 3.0, "min_k": 0, "max_k": 15},
    "sae": _SAE_META,
}
# A deterministic warm-start the stubbed DiffMean returns. Its ids are EVEN ⊆ V so the
# COMPACTED encoder conditions on them without OOV: the stub mimics the real
# feature_diffmean_bag(allowed_ids=...), which V-restricts its output (#720 T2).
_WARM_BAG = {4: 0, 12: 0, 22: 1, 30: 1}


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention has no CPU lowering/backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny-model + data builders
# ─────────────────────────────────────────────────────────────────────────────


def _build_model(monkeypatch: pytest.MonkeyPatch, *, feature_vocab: FeatureVocab | None) -> Any:
    """A tiny CPU/float32 encdec, optionally COMPACTED via ``feature_vocab``.

    Mirrors ``build_tiny_model``'s shapes but threads ``feature_vocab`` through the ctor
    (``build_tiny_model`` exposes no such knob), so the compacted and uncompacted models
    are otherwise shape-identical and accept the SAME members + bag.  ``COMPUTE_DTYPE`` is
    pinned to float32 across the four modules that read it so the CPU forward is clean.
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
    set_runtime_attribute(model, "_loaded_meta_data", {"user_config": _USER_CONFIG})
    return model


def _make_prefix(idx: int = 0) -> BpbAttackPrefix:
    """A prefix whose tokens round-trip from its text under the stub nanochat tokenizer."""
    tok = StubNanochatTokenizer()
    prefix_text = f"10 11 {12 + idx} 13"
    return BpbAttackPrefix(
        core_prefix_id=f"c7_positive_p{idx}",
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        prefix_text=prefix_text,
        prefix_hash=f"h7positive{idx}",
        prefix_tokens=encode_prefix(prefix_text, tok),
        concept_rich_sample_ids=["r0"],
        concept_avoiding_sample_ids=["a0"],
        concept_rich_output_tokens=[[20, 21, 22, 23]],
        concept_avoiding_output_tokens=[[30, 31, 32, 33]],
    )


def _segments_for(prefix: BpbAttackPrefix, tokenizer: Any) -> tuple[ContextFeatureSegment, ...]:
    """One EMPTY-feature context segment spanning the decoded prefix (2-slot layout)."""
    char_end = len(tokenizer.decode(list(prefix.prefix_tokens)))
    return (
        ContextFeatureSegment(feature_ids=(), buckets=(), char_start=0, char_end=char_end),
    )


def _members(prefixes: list[BpbAttackPrefix]) -> list[Any]:
    tok = StubNanochatTokenizer()
    out: list[Any] = []
    for prefix in prefixes:
        out.extend(
            build_class_margin_members(
                prefix,
                prefix_segments=_segments_for(prefix, tok),
                tokenizer=tok,
                max_seq_len=_MAX_SEQ_LEN,
                device=_CPU,
            )
        )
    return out


def _stub_feature_diffmean_bag(*_args: Any, **_kwargs: Any) -> FeatureConditioning:
    """Deterministic warm-start seam (the real DiffMean SAE forward is not under test)."""
    return FeatureConditioning.from_dict(dict(_WARM_BAG))


def _make_cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        neuronpedia_source="8-gemmascope-res-16k",
        neuronpedia_model_id="gemma-2-2b",
        base_methods=None,
        feature_dir="",
        extraction_tokenizer_id="google/gemma-2-2b",
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=(0.0,),
            split_ratio=0.6,
            split_seed=0,
            max_steps=8,
            minibatch_size=8,
            master_seed=0,
            mean_g_steer_batch=16,
            max_forward_batch=8,
        ),
    )


def _context_map(
    prefixes: list[BpbAttackPrefix], tokenizer: Any
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    return {p.prefix_hash: _segments_for(p, tokenizer) for p in prefixes}


# ─────────────────────────────────────────────────────────────────────────────
# 1) Uncompacted no-op — the real forward runs
# ─────────────────────────────────────────────────────────────────────────────


def test_uncompacted_mean_g_steer_runs_and_returns_finite_g_steer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``feature_vocab=None`` ⇒ the guard is a strict no-op and the measurement runs."""
    model = _build_model(monkeypatch, feature_vocab=None)
    assert model.encoder.feature_vocab is None, "sanity: this model is NOT compacted"
    members = _members([_make_prefix()])
    assert members, "fixture must produce >= 1 class-margin member"

    g = mean_g_steer(model, members, _BAG, mode=SteeringMode.POSITIVE, device=_CPU)

    assert isinstance(g, float)
    assert g == g, "uncompacted g_steer must be finite (the forward really ran)"


# ─────────────────────────────────────────────────────────────────────────────
# 2) Compacted + V-restricted bag — the measurement RUNS (#720)
# ─────────────────────────────────────────────────────────────────────────────


def test_compacted_mean_g_steer_runs_with_in_vocab_bag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compacted encoder no longer refuses ``mean_g_steer``: with a V-restricted bag
    (``_BAG`` ⊆ V) it RUNS and returns a finite g_steer (#720 PR-2 T3).

    Direct proof the lifted guard was safe: the SAME call that raised before now completes
    through the compacted encoder's ``_to_compact`` (every bag id has a compact row).
    """
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    assert model.encoder.feature_vocab is not None, "sanity: this model IS compacted"
    members = _members([_make_prefix()])
    assert members, "fixture must produce >= 1 class-margin member"

    g = mean_g_steer(model, members, _BAG, mode=SteeringMode.POSITIVE, device=_CPU)

    assert isinstance(g, float)
    assert g == g, "compacted g_steer must be finite (the forward really ran)"


def test_out_of_vocab_bag_still_trips_the_encoder_oov_backstop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lifted guard is not a safety regression: conditioning a compacted encoder on an
    OUT-of-V id in a real slot still fails loud at the encoder's own ``_to_compact`` OOV
    assert (encoder.py). The derive relies on V-restricting upstream (T2); this proves the
    last-line backstop is still there if an un-restricted id ever reached the forward."""
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    assert model.encoder.feature_vocab is not None, "sanity: this model IS compacted"
    members = _members([_make_prefix()])

    with pytest.raises(RuntimeError):
        mean_g_steer(model, members, _OOV_BAG, mode=SteeringMode.POSITIVE, device=_CPU)


# ─────────────────────────────────────────────────────────────────────────────
# 2b) Non-empty MIXED context — the restriction runs through a REAL forward
# ─────────────────────────────────────────────────────────────────────────────
#
# Every fixture above (`_segments_for`) builds the member's context segment EMPTY
# (``feature_ids=(), buckets=()``), so ``restrict_segment_conditions_to_vocab`` is a
# structural no-op there and the encoder only ever forwards ``n_real == 0`` context
# slots. A segment-count/order/padding bug in the non-empty-context restriction path
# would pass every test above. This builds a member whose context segment is genuinely
# NON-EMPTY and MIXES in-V and out-of-V original ids, then runs the real compacted
# forward path (``mean_g_steer``, the derives' shared choke point) through it.


def _mixed_context_segments_for(
    prefix: BpbAttackPrefix, tokenizer: Any
) -> tuple[ContextFeatureSegment, ...]:
    """One NON-EMPTY context segment mixing IN-V (even) and OUT-of-V (odd) original
    feature ids. Even ids are in ``_ALLOWED``; odd ids are in-range (< ``_N_SAE``) but NOT
    in V -- the shape that trips ``_to_compact``'s OOV assert in a real slot if the
    restriction is bypassed. Buckets are valid for ``_NUM_BUCKETS=2`` (0 or 1)."""
    char_end = len(tokenizer.decode(list(prefix.prefix_tokens)))
    return (
        ContextFeatureSegment(
            feature_ids=(4, 5, 12, 13), buckets=(0, 1, 0, 1), char_start=0, char_end=char_end
        ),
    )


def test_nonempty_mixed_context_is_restricted_before_a_real_compacted_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member built from a NON-EMPTY, MIXED (in-V + out-of-V) context segment: the
    restriction must drop the out-of-V ids -- so ``mean_g_steer``'s real compacted forward
    does NOT hit the encoder's ``_to_compact`` OOV device-assert -- while keeping the
    in-V ones, so this is a genuine ``n_real > 0`` forward, not another vacuous
    empty-context run.
    """
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    prefix = _make_prefix()
    tok = StubNanochatTokenizer()
    raw_segments = _mixed_context_segments_for(prefix, tok)

    raw_ids = set(raw_segments[0].feature_ids)
    in_v_raw = raw_ids & set(_ALLOWED)
    out_of_v_raw = raw_ids - set(_ALLOWED)
    assert in_v_raw, "fixture must contain >= 1 in-V id (else the kept side is vacuous)"
    assert out_of_v_raw, "fixture must contain >= 1 out-of-V id (else the drop side is vacuous)"

    members = build_class_margin_members(
        prefix,
        prefix_segments=raw_segments,
        tokenizer=tok,
        max_seq_len=_MAX_SEQ_LEN,
        device=_CPU,
        allowed_ids=feature_vocab_allowed_ids(model),
    )
    assert members, "fixture must produce >= 1 class-margin member"

    # (1) the restriction actually ran: the built member's context bag == the in-V subset
    #     of the raw segment -- the out-of-V ids were dropped, the in-V ones kept.
    context_bag = members[0].rich.segment_conditions[0]
    restricted_ids = {fid for fid, _bucket in context_bag.entries}
    assert restricted_ids == in_v_raw
    assert restricted_ids.isdisjoint(out_of_v_raw)

    # (2) the real compacted forward (mean_g_steer) RUNS through this non-empty,
    #     restricted context and returns finite -- it does NOT hit the encoder's
    #     _to_compact OOV device-assert.
    g = mean_g_steer(model, members, _BAG, mode=SteeringMode.POSITIVE, device=_CPU)
    assert isinstance(g, float)
    assert g == g, "g_steer must be finite: the forward really ran with a non-empty context"


# ─────────────────────────────────────────────────────────────────────────────
# 3) The shared choke point — proving fair AND roofline are both covered
# ─────────────────────────────────────────────────────────────────────────────


def test_group_g_steer_is_the_shared_mean_g_steer_choke_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both derives measure g_steer ONLY through ``_group_g_steer`` → ``mean_g_steer``.

    ``_group_g_steer`` is a ``functools.partial`` over ``mean_g_steer``; fair
    (``build_ours_gcg_bags_fair`` line 732) and roofline
    (``build_ours_gcg_bags_roofline`` line 909) each build exactly one and route every
    measurement through it.  So the single guard on ``mean_g_steer`` covers both — this
    pins that the partial really wraps ``mean_g_steer`` (not a copy that could drift).
    """
    model = _build_model(monkeypatch, feature_vocab=None)
    group_g_steer = gcg_bags._group_g_steer(
        model, OursGcgConfig(), SteeringMode.POSITIVE, _CPU
    )
    assert isinstance(group_g_steer, partial)
    assert group_g_steer.func is mean_g_steer


def test_group_g_steer_partial_runs_on_compacted_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EXACT indirection both derives use (``_group_g_steer`` partial) RUNS on a
    compacted model with a V-restricted bag.

    Behavioural half of the coverage proof: calling the partial the way both derives call
    ``group_g_steer(members, bag)`` now completes on a compacted model, returning a finite
    g_steer — so neither family is blocked.
    """
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    group_g_steer = gcg_bags._group_g_steer(
        model, OursGcgConfig(), SteeringMode.POSITIVE, _CPU
    )
    g = group_g_steer(_members([_make_prefix()]), _BAG)
    assert isinstance(g, float) and g == g, "the partial must run and return a finite g_steer"


# ─────────────────────────────────────────────────────────────────────────────
# 4)/5) Each named derive completes end-to-end on a compacted checkpoint, bags ⊆ V
# ─────────────────────────────────────────────────────────────────────────────


def test_fair_derive_runs_on_compacted_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``build_ours_gcg_bags_fair`` completes on a compacted checkpoint and writes a bag ⊆ V.

    The warm-start seam is stubbed to return a V-restricted bag (mirroring the real
    ``feature_diffmean_bag(allowed_ids=...)``); the universal optimizer proposes only ids in
    V (#720 T1). So the derive conditions the encoder on in-V ids only, runs the warm +
    optimize + held-out ``group_g_steer`` measurements, and writes a bag whose feature ids
    all lie in V — the SAME path that raised a clean ``NotImplementedError`` before.
    """
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", _stub_feature_diffmean_bag)
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    prefixes = [_make_prefix(i) for i in range(5)]
    tok = StubNanochatTokenizer()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair_compact")

    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=paths,
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )

    bag_path = paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, "fair")
    assert bag_path.exists(), f"fair derive must write a bag: {bag_path}"
    bag_dict, _metadata = read_ours_gcg_bag(bag_path)
    assert bag_dict, "refined fair bag is unexpectedly empty"
    allowed_set = set(_ALLOWED)
    assert set(bag_dict) <= allowed_set, (
        f"fair bag ids must all lie in V={sorted(allowed_set)}; got {sorted(bag_dict)}"
    )


def test_roofline_derive_runs_on_compacted_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``build_ours_gcg_bags_roofline`` completes on a compacted checkpoint, per-prefix bags ⊆ V.

    The roofline sibling of the fair test: same shared ``group_g_steer`` measurement, now
    RUN (not refused) per held-out prefix, each writing a bag whose feature ids lie in V.
    """
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", _stub_feature_diffmean_bag)
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    prefixes = [_make_prefix(i) for i in range(5)]
    tok = StubNanochatTokenizer()
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline_compact")

    build_ours_gcg_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=paths,
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )

    _optimize, held_out = split_prefixes_for_concept(prefixes, ratio=0.6, seed=0)
    assert held_out, "fixture must produce >= 1 held-out prefix"
    allowed_set = set(_ALLOWED)
    for prefix in held_out:
        bag_path = paths.ours_gcg_bag(
            7, SteeringMode.POSITIVE, 0.0, "roofline", prefix_id=prefix.core_prefix_id
        )
        assert bag_path.exists(), f"roofline derive must write a per-prefix bag: {bag_path}"
        bag_dict, _metadata = read_ours_gcg_bag(bag_path)
        assert bag_dict, "roofline bag is unexpectedly empty"
        assert set(bag_dict) <= allowed_set, (
            f"roofline bag ids must all lie in V; got {sorted(bag_dict)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6) Warm-bag threading — allowed_ids reaches feature_diffmean_bag (#720 PR-2 T2)
# ─────────────────────────────────────────────────────────────────────────────
#
# The derive now COMPLETES on a compacted model, so it threads
# ``allowed_ids=feature_vocab_allowed_ids(model)`` into the warm ``feature_diffmean_bag``
# call and runs to the end. These tests CAPTURE that kwarg (checked against the exact
# tensor the encoder itself uses) as the derive runs. "Uncompacted unchanged" is pinned
# separately by the full-completion derive tests in ``test_gcg_bags_fair.py`` /
# ``test_gcg_bags_roofline.py`` (their stubs assert ``allowed_ids is None``).


def _capturing_diffmean_stub() -> tuple[Callable[..., FeatureConditioning], dict[str, Any]]:
    """A deterministic warm-start stub that RECORDS its ``allowed_ids`` kwarg."""
    captured: dict[str, Any] = {}

    def _stub(*_args: Any, allowed_ids: Any = None, **_kwargs: Any) -> FeatureConditioning:
        captured["allowed_ids"] = allowed_ids
        return FeatureConditioning.from_dict(dict(_WARM_BAG))

    return _stub, captured


def test_fair_derive_threads_allowed_ids_into_warm_diffmean_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On a COMPACTED model, ``build_ours_gcg_bags_fair`` completes and its warm
    ``feature_diffmean_bag`` call receives ``allowed_ids`` == the model's OWN
    V -- checked directly against ``feature_vocab_allowed_ids(model)``, the
    exact tensor the encoder itself uses (not a hand-reconstructed guess)."""
    stub, captured = _capturing_diffmean_stub()
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", stub)
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    prefixes = [_make_prefix(i) for i in range(5)]
    tok = StubNanochatTokenizer()

    build_ours_gcg_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair_compact_warm"),
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )

    assert "allowed_ids" in captured, "feature_diffmean_bag must have been called"
    allowed_ids = captured["allowed_ids"]
    assert allowed_ids is not None
    expected = feature_vocab_allowed_ids(model)
    assert expected is not None
    assert t.equal(allowed_ids, expected)
    assert t.equal(allowed_ids, t.tensor(_ALLOWED, dtype=t.int64))


def test_roofline_derive_threads_allowed_ids_into_warm_diffmean_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The roofline sibling: same threading proof, reached on the FIRST
    held-out prefix's in-sample warm-start call as the derive completes."""
    stub, captured = _capturing_diffmean_stub()
    monkeypatch.setattr(gcg_bags, "feature_diffmean_bag", stub)
    model = _build_model(monkeypatch, feature_vocab=_VOCAB)
    prefixes = [_make_prefix(i) for i in range(5)]
    tok = StubNanochatTokenizer()

    build_ours_gcg_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=StubBaseModel(),
        sae=object(),
        percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
        cfg=_make_cfg(),
        paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline_compact_warm"),
        device=_CPU,
        tokenizer=tok,
        context_segments_by_prefix_hash=_context_map(prefixes, tok),
    )

    assert "allowed_ids" in captured, "feature_diffmean_bag must have been called"
    allowed_ids = captured["allowed_ids"]
    assert allowed_ids is not None
    expected = feature_vocab_allowed_ids(model)
    assert expected is not None
    assert t.equal(allowed_ids, expected)
