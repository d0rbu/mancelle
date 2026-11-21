"""Stub-model orchestration coverage for the roofline general-val scoring path.

Closes the gap the #642 review flagged: the pure roofline helpers are unit
tested, but the model-backed orchestration (`_score_roofline_combo` ->
`_roofline_bag_for_doc` -> derive/optimize/persist -> `apply_additive_counter`
-> `score_triple` -> row emission) had no test. This drives it end-to-end with a
deterministic stub encdec model and stubbed leaf objectives, so the routing,
the DiffMean-vs-GCG bag branch, the cache miss->write->hit path, the fail-loud
raises, and the per-doc footprint (one score row per doc) are all exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv import general_val
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val import (
    _doc_to_scoring_input,
    _GeneralValDoc,
    _InterventionCombo,
    _RooflineGeneralValState,
    _score_roofline_combo,
)
from steering.eval.axbench_bpb_adv.general_val_roofline import GeneralValSignalBudget
from steering.eval.axbench_bpb_adv.intervention_build import (
    AdditiveCounterNoOp,
    AdditiveCounterNoOpReason,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag
from steering.eval.axbench_bpb_adv.precomputed_bag_family import PrecomputedBagFamily
from steering.eval.target_conditioning import ContextFeatureSegment

_CORPUS = "nd=2;nb=8;h=deadbeefdeadbeef"
_CONFIG_METADATA = {
    "extraction_threshold": "0.0",
    "neuronpedia_model_id": "gemma-2-2b",
    "context_cache_num_buckets": "2",
    "bucket_boundaries_sha256": "abc123",
}


class _FakeTokenizer:
    """One-char-per-token, matching the roofline unit tests' fake tokenizer."""

    def decode(self, token_ids: Any) -> str:
        return "".join(chr(65 + int(tid) % 26) for tid in token_ids)


class _CountingEncdecModel:
    """Deterministic encdec stub: loss at each position = 0.1 * (# conditioning
    entries routed to that position's encoder slot). No parameters; exercises the
    real score_triple encdec kernel (padding, seg_map scatter, BPB accumulation).
    """

    def __init__(self) -> None:
        # score_triple's teardown fingerprint iterates transformer.h; an empty
        # block list makes the fingerprint trivially stable (roofline installs no
        # residual hook — Intervention.ours consumes its payload in the encoder).
        self.transformer = SimpleNamespace(h=[])
        # _pad_encoder_inputs reads `.feature_vocab` (via
        # `feature_vocab_allowed_ids`) to V-restrict context-cache features;
        # `None` == uncompacted, matching this stub's unrestricted behaviour.
        self.feature_vocab = None
        # The real encdec loader attaches these fields; roofline general-val GCG
        # derives warm-start partition caps from the same metadata before cache
        # reuse or optimization.
        self.encoder = SimpleNamespace(config=SimpleNamespace(num_buckets=2))
        self._loaded_meta_data = {
            "user_config": {
                "feature_selection": {"distribution": "geometric", "mean": 3.0},
                "negative_selection": {"distribution": "geometric", "mean": 2.0},
            }
        }

    def eval(self) -> _CountingEncdecModel:
        return self

    def train(self, mode: bool = True) -> _CountingEncdecModel:
        return self

    def __call__(
        self,
        dec_input: t.Tensor,
        dec_target: t.Tensor,
        *,
        enc_feature_ids: t.Tensor,
        enc_magnitudes: t.Tensor,
        enc_mask: t.Tensor,
        seg_map: t.Tensor,
        loss_reduction: str,
    ) -> t.Tensor:
        assert loss_reduction == "none"
        counts = enc_mask.sum(dim=-1).to(t.float32)  # (B, n_seg)
        per_pos = t.gather(counts, 1, seg_map)  # (B, T-1)
        return (0.1 * per_pos + 0.05).reshape(-1)  # strictly positive, finite


def _doc(sample_id: str, prefix_tokens: list[int], output_tokens: list[int]) -> _GeneralValDoc:
    return _GeneralValDoc(
        sample_id=sample_id,
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_text=_FakeTokenizer().decode(prefix_tokens),
        output_byte_weights=t.ones(len(output_tokens), dtype=t.long),
    )


def _segment(prefix_tokens: list[int]) -> ContextFeatureSegment:
    return ContextFeatureSegment(
        feature_ids=(5,),
        buckets=(1,),
        char_start=0,
        char_end=len(_FakeTokenizer().decode(prefix_tokens)),
    )


def _combo(
    *,
    method_name: str,
    coefficient_str: str,
    target_model: str = "encdec",
    adversarial_method: str = "base",
    adversarial_strength: float = 0.0,
) -> _InterventionCombo:
    return _InterventionCombo(
        target_model=target_model,
        method_name=method_name,
        mode="positive",
        concept_id=1,
        coefficient_str=coefficient_str,
        adversarial_method=adversarial_method,
        adversarial_strength=adversarial_strength,
        strength_index=0,
    )


def _fixtures() -> dict[str, Any]:
    docs = [
        _doc("general_val_0", [0, 1, 2], [3, 4]),
        _doc("general_val_1", [0, 5, 6], [7, 8]),
    ]
    segments_by_sample = {d.sample_id: (_segment(d.prefix_tokens),) for d in docs}
    samples_by_id = {
        d.sample_id: _doc_to_scoring_input(
            d, segments=segments_by_sample[d.sample_id], max_seq_len=64
        )
        for d in docs
    }
    warm_bags = {d.sample_id: FeatureConditioning.from_dict({10 + i: 1}) for i, d in enumerate(docs)}
    return {
        "docs": docs,
        "segments_by_sample": segments_by_sample,
        "samples_by_id": samples_by_id,
        "warm_bags": warm_bags,
    }


def _roofline_state(fixtures: dict[str, Any]) -> _RooflineGeneralValState:
    state = _RooflineGeneralValState.for_budget(
        GeneralValSignalBudget(
            k_pos=3,
            k_neg=2,
            positive_bucket=0,
            negative_bucket=1,
        )
    )
    state.real_bags.update(fixtures["warm_bags"])
    state.bag_config_metadata.update(_CONFIG_METADATA)
    return state


def _call_score(
    *,
    combo: _InterventionCombo,
    family: PrecomputedBagFamily,
    paths: AxBenchBpbAdvPaths,
    fixtures: dict[str, Any],
    cfg: AxBenchBpbAdvConfig,
    roofline_state: _RooflineGeneralValState | None = None,
) -> list[Any]:
    return _score_roofline_combo(
        combo=combo,
        family=family,
        concept=cast(Any, SimpleNamespace(concept_id=1, feature_index=0)),
        method=SimpleNamespace(name=combo.method_name, mode=SteeringMode.POSITIVE),
        docs=fixtures["docs"],
        samples_by_id=fixtures["samples_by_id"],
        segments_by_sample=fixtures["segments_by_sample"],
        roofline_state=(
            roofline_state if roofline_state is not None else _roofline_state(fixtures)
        ),
        target_bundle=TargetModelBundle(
            model=_CountingEncdecModel(),
            tokenizer=_FakeTokenizer(),
            max_seq_len=64,
            target_name="encdec",
        ),
        paths=paths,
        source_paths=cast(Any, SimpleNamespace()),
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        cfg=cfg,
        counter_direction_family="diffmean",
        apply_max_act_scale=False,
        corpus_signature=_CORPUS,
        lora=None,
    )


def test_roofline_state_bundles_budget_real_bags_and_config_metadata() -> None:
    fx = _fixtures()
    state = _roofline_state(fx)

    assert state.signal_budget == GeneralValSignalBudget(
        k_pos=3,
        k_neg=2,
        positive_bucket=0,
        negative_bucket=1,
    )
    assert state.real_bags == fx["warm_bags"]
    assert state.bag_config_metadata == _CONFIG_METADATA


def test_roofline_state_allocates_independent_mutable_maps() -> None:
    budget = GeneralValSignalBudget(
        k_pos=3,
        k_neg=2,
        positive_bucket=0,
        negative_bucket=1,
    )
    state_a = _RooflineGeneralValState.for_budget(budget)
    state_b = _RooflineGeneralValState.for_budget(budget)

    state_a.real_bags["general_val_0"] = FeatureConditioning.from_dict({1: 0})
    state_a.bag_config_metadata["bucket_boundaries_sha256"] = "abc"

    assert state_b.real_bags == {}
    assert state_b.bag_config_metadata == {}
    assert state_a.real_bags is not state_b.real_bags
    assert state_a.bag_config_metadata is not state_b.bag_config_metadata


@pytest.fixture()
def _cfg() -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/s", output_root="/tmp/o", feature_dir="/tmp/f",
        feature_statistics_path="/tmp/stats",
    )
    cfg.ours_gcg = OursGcgConfig(max_steps=8, minibatch_size=16, master_seed=0, max_forward_batch=8)
    cfg.diffmean_and_probe_layer = 6
    cfg.device_batch_size = 4
    return cfg


def _stub_leaf_objectives(
    monkeypatch: pytest.MonkeyPatch,
    optimize_calls: list[int],
    objective_calls: list[int] | None = None,
) -> None:
    def _fake_objective(*_args: Any, **_kwargs: Any) -> float:
        if objective_calls is not None:
            objective_calls.append(1)
        return 0.5

    monkeypatch.setattr(general_val, "penalized_lm_objective_value", _fake_objective)

    def _fake_optimize(model: Any, member: Any, *, warm: FeatureConditioning, **kw: Any) -> FeatureConditioning:
        optimize_calls.append(1)
        return warm.with_added(999, 0)

    monkeypatch.setattr(general_val, "optimize_general_val_doc_bag", _fake_optimize)


def test_diffmean_roofline_scores_every_doc_with_warm_bag(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    optimize_calls: list[int] = []
    objective_calls: list[int] = []
    _stub_leaf_objectives(monkeypatch, optimize_calls, objective_calls)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")

    rows = _call_score(
        combo=combo, family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        paths=paths, fixtures=fx, cfg=_cfg,
    )

    # Footprint: exactly one BPB score row per doc.
    assert len(rows) == len(fx["docs"])
    assert all(r.bpb_intervened > 0.0 and r.n_output_bytes > 0 for r in rows)
    # DiffMean roofline does NOT optimize; the persisted bag is the warm signal.
    assert optimize_calls == []
    # It also skips the metadata-only target-model objective forward.
    assert objective_calls == []
    for d in fx["docs"]:
        bag, meta = read_ours_gcg_bag(
            paths.general_val_bag(d.sample_id, PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN, 0.0)
        )
        assert bag == dict(fx["warm_bags"][d.sample_id].entries)
        assert meta["general_val_corpus"] == _CORPUS  # corpus threaded into metadata
        assert meta["signal_policy"] == "top_k_by_partition_full_percentile_v1"
        assert meta["signal_k_pos"] == "3"
        assert meta["signal_k_neg"] == "2"
        assert meta["positive_bucket"] == "0"
        assert meta["negative_bucket"] == "1"
        for key, expected in _CONFIG_METADATA.items():
            assert meta[key] == expected


def test_gcg_roofline_optimizes_persists_and_reuses_bags(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    optimize_calls: list[int] = []
    objective_calls: list[int] = []
    _stub_leaf_objectives(monkeypatch, optimize_calls, objective_calls)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(method_name="ours_gcg_roofline_encdec", coefficient_str="0.0")

    rows_1 = _call_score(
        combo=combo, family=PrecomputedBagFamily.ROOFLINE_GCG, paths=paths, fixtures=fx, cfg=_cfg
    )
    assert len(rows_1) == len(fx["docs"])
    n_first = len(optimize_calls)
    assert n_first == len(fx["docs"])  # one optimize per doc on the cache-miss run
    assert len(objective_calls) == 2 * len(fx["docs"])  # warm + optimized objective per doc
    for d in fx["docs"]:
        bag, meta = read_ours_gcg_bag(
            paths.general_val_bag(d.sample_id, PrecomputedBagFamily.ROOFLINE_GCG, 0.0)
        )
        assert 999 in bag  # the (stubbed) optimizer's edit was persisted
        assert meta["signal_policy"] == "top_k_by_partition_full_percentile_v1"
        assert meta["signal_k_pos"] == "3"
        assert meta["signal_k_neg"] == "2"
        assert meta["positive_bucket"] == "0"
        assert meta["negative_bucket"] == "1"
        assert meta["max_k_pos"] == "100"
        assert meta["max_k_neg"] == "100"
        for key, expected in _CONFIG_METADATA.items():
            assert meta[key] == expected

    # Second call: bags are read from disk, optimizer NOT re-invoked; same scores.
    rows_2 = _call_score(
        combo=combo, family=PrecomputedBagFamily.ROOFLINE_GCG, paths=paths, fixtures=fx, cfg=_cfg
    )
    assert len(optimize_calls) == n_first  # no new optimize calls (cache hit)
    assert len(objective_calls) == 2 * len(fx["docs"])  # no objective calls on cache hit
    assert [r.bpb_intervened for r in rows_2] == [r.bpb_intervened for r in rows_1]


def test_roofline_additive_counter_resolved_once_per_combo(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    _stub_leaf_objectives(monkeypatch, [])
    calls: list[dict[str, Any]] = []

    def _fake_resolve(**kwargs: Any) -> AdditiveCounterNoOp:
        calls.append(kwargs)
        return AdditiveCounterNoOp(AdditiveCounterNoOpReason.NOT_ADDITIVE_DIRECTION)

    monkeypatch.setattr(general_val, "resolve_additive_counter", _fake_resolve)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(
        method_name="feature_diffmean_roofline_encdec",
        coefficient_str="None",
        adversarial_method="additive_direction",
        adversarial_strength=1.0,
    )

    rows = _call_score(
        combo=combo,
        family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        paths=paths,
        fixtures=fx,
        cfg=_cfg,
    )

    assert len(rows) == len(fx["docs"])
    assert len(calls) == 1
    assert calls[0]["adversarial_method"] == "additive_direction"
    assert calls[0]["adversarial_strength"] == 1.0


def test_score_roofline_combo_rejects_non_encdec_target(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    _stub_leaf_objectives(monkeypatch, [])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(
        method_name="feature_diffmean_roofline_encdec", coefficient_str="None",
        target_model="decoder_only",
    )
    with pytest.raises(AxBenchBpbAdvError, match="only encdec targets"):
        _call_score(
            combo=combo, family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            paths=paths, fixtures=fx, cfg=_cfg,
        )


def test_score_roofline_combo_raises_on_missing_scored_sample(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    _stub_leaf_objectives(monkeypatch, [])
    # Drop one doc from the scored sample set -> the per-doc guard must raise.
    del fx["samples_by_id"]["general_val_1"]
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")
    with pytest.raises(AxBenchBpbAdvError, match="missing from the encdec scored sample set"):
        _call_score(
            combo=combo, family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            paths=paths, fixtures=fx, cfg=_cfg,
        )


def test_score_roofline_combo_raises_on_missing_real_bag_state(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    fx = _fixtures()
    _stub_leaf_objectives(monkeypatch, [])
    state = _roofline_state(fx)
    del state.real_bags["general_val_1"]
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="r")
    combo = _combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")

    with pytest.raises(AxBenchBpbAdvError, match="missing real output-window warm bag"):
        _call_score(
            combo=combo,
            family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            paths=paths,
            fixtures=fx,
            cfg=_cfg,
            roofline_state=state,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Context-bag restriction wiring (issue #720 PR-2 Task 2): `_roofline_bag_for_doc`
# never passed `allowed_ids` to `build_general_val_doc_member`, so a compacted
# derive would condition the FIXED context segments on UN-restricted
# (original-SAE-id) ids. `build_general_val_doc_member` itself already restricts
# correctly when given `allowed_ids` (see
# tests/axbench_bpb_adv/unit/test_general_val_roofline_vocab.py) -- this section
# proves the CALLER actually threads it, through the REAL, un-stubbed
# `_roofline_bag_for_doc`. Family=ROOFLINE_FEATURE_DIFFMEAN reaches the SAME
# member-build call as ROOFLINE_GCG (both branch AFTER it) while never reaching
# the unrelated, T1-owned, T3-deferred `penalized_lm_objective_value` compacted
# guard.
# ─────────────────────────────────────────────────────────────────────────────


def _compacted_target_model(allowed_ids: t.Tensor, *, fingerprint: str) -> Any:
    """A compacted target-model stub carrying every attribute EITHER roofline family
    reads: ``feature_vocab``/``encoder.feature_vocab_ids`` (``feature_vocab_allowed_ids``,
    both families), plus ``encoder.config.num_buckets``/``_loaded_meta_data``
    (``partition_bounds_from_model``, read ONLY by the ROOFLINE_GCG branch's
    ``general_val_doc_gcg_config`` -- harmless, unread extras for the DiffMean tests).
    Metadata mirrors ``_CountingEncdecModel``'s (this file's other model stub)."""
    return SimpleNamespace(
        feature_vocab=SimpleNamespace(fingerprint=lambda: fingerprint),
        encoder=SimpleNamespace(
            feature_vocab_ids=allowed_ids,
            config=SimpleNamespace(num_buckets=2),
        ),
        _loaded_meta_data={
            "user_config": {
                "feature_selection": {"distribution": "geometric", "mean": 3.0},
                "negative_selection": {"distribution": "geometric", "mean": 2.0},
            }
        },
    )


def _spy_on_member_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Wrap the REAL `build_general_val_doc_member` so it still runs for real,
    while capturing every call's kwargs and the member it built."""
    real_builder = general_val.build_general_val_doc_member
    members: list[Any] = []
    calls: list[dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        member = real_builder(*args, **kwargs)
        members.append(member)
        return member

    monkeypatch.setattr(general_val, "build_general_val_doc_member", _spy)
    return members, calls


def _member_context_ids(member: Any) -> set[int]:
    return {fid for bag in member.segment_conditions[:-1] for fid, _bucket in bag.entries}


def test_roofline_bag_for_doc_restricts_member_context_to_compacted_vocab(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    """The REAL `_roofline_bag_for_doc` -> `build_general_val_doc_member` call
    must restrict the built member's FIXED context bags to a compacted
    checkpoint's V.

    Mutation check: deleting the `allowed_ids=...` kwarg this task adds to the
    `build_general_val_doc_member` call reproduces today's dead-wiring bug --
    id 99 (outside V) survives into the built member -- and REDs this test.
    """
    allowed_ids = t.tensor([1, 2], dtype=t.int64)
    model = _compacted_target_model(allowed_ids, fingerprint="fp-general-val-ctx")
    target_bundle = TargetModelBundle(
        model=model, tokenizer=_FakeTokenizer(), max_seq_len=64, target_name="encdec"
    )
    doc = _doc("general_val_ctx_0", [0, 1, 2], [3, 4])
    seg = ContextFeatureSegment(
        feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3
    )
    warm = FeatureConditioning.from_dict({9: 0})
    state = _RooflineGeneralValState.for_budget(
        GeneralValSignalBudget(k_pos=3, k_neg=2, positive_bucket=0, negative_bucket=1)
    )
    state.real_bags[doc.sample_id] = warm
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="general-val-ctx-vocab")
    combo = _combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")

    members, calls = _spy_on_member_builder(monkeypatch)

    bag = general_val._roofline_bag_for_doc(
        combo=combo,
        family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        doc=doc,
        segments=(seg,),
        roofline_state=state,
        target_bundle=target_bundle,
        paths=paths,
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        corpus_signature=_CORPUS,
        cfg=_cfg,
    )

    assert bag is warm  # DiffMean family: the output bag itself is untouched
    assert len(members) == 1
    assert calls[0].get("allowed_ids") is allowed_ids  # the model's OWN buffer
    context_ids = _member_context_ids(members[0])
    assert context_ids == {1, 2}
    assert 99 not in context_ids


def test_roofline_bag_for_doc_restricts_member_context_to_compacted_vocab_for_gcg_family(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    """The GCG sibling of the DiffMean test above: no existing test drives the REAL
    `_roofline_bag_for_doc` with `family=ROOFLINE_GCG` on a compacted model. The DiffMean
    test above reaches the SAME `build_general_val_doc_member` call but returns (`bag =
    warm`) before ever entering the GCG-only branch (`general_val_doc_gcg_config` before
    the member build; `penalized_lm_objective_value` / `optimize_general_val_doc_bag`
    after it) -- so a bug that makes the context restriction depend on `family` (e.g.
    threading `allowed_ids` only when `family is ROOFLINE_FEATURE_DIFFMEAN`) would pass
    every current test and only show up here.

    The heavy optimizer/objective calls are stubbed (`_stub_leaf_objectives`, as the
    DiffMean-vs-GCG routing test above it does) -- this test targets the member-build
    wiring, not the optimizer itself; `optimize_feature_conditioning_universal`'s own
    V-restriction is proven separately
    (``tests/steering/feature_gcg/integration/test_compacted_optimizer_equivalence.py``).

    Mutation check: making `_roofline_bag_for_doc`'s `allowed_ids=...` kwarg to
    `build_general_val_doc_member` family-conditional (e.g. `allowed_ids =
    feature_vocab_allowed_ids(model) if family is ROOFLINE_FEATURE_DIFFMEAN else None`)
    REDs this test -- id 99 (outside V) would survive into the built member.
    """
    _stub_leaf_objectives(monkeypatch, [])
    allowed_ids = t.tensor([1, 2], dtype=t.int64)
    model = _compacted_target_model(allowed_ids, fingerprint="fp-general-val-ctx-gcg")
    target_bundle = TargetModelBundle(
        model=model, tokenizer=_FakeTokenizer(), max_seq_len=64, target_name="encdec"
    )
    doc = _doc("general_val_ctx_gcg_0", [0, 1, 2], [3, 4])
    seg = ContextFeatureSegment(
        feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3
    )
    warm = FeatureConditioning.from_dict({9: 0})
    state = _RooflineGeneralValState.for_budget(
        GeneralValSignalBudget(k_pos=3, k_neg=2, positive_bucket=0, negative_bucket=1)
    )
    state.real_bags[doc.sample_id] = warm
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="general-val-ctx-vocab-gcg")
    combo = _combo(method_name="ours_gcg_roofline_encdec", coefficient_str="0.0")

    members, calls = _spy_on_member_builder(monkeypatch)

    general_val._roofline_bag_for_doc(
        combo=combo,
        family=PrecomputedBagFamily.ROOFLINE_GCG,
        doc=doc,
        segments=(seg,),
        roofline_state=state,
        target_bundle=target_bundle,
        paths=paths,
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        corpus_signature=_CORPUS,
        cfg=_cfg,
    )

    assert len(members) == 1
    assert calls[0].get("allowed_ids") is allowed_ids  # the model's OWN buffer, GCG family too
    context_ids = _member_context_ids(members[0])
    assert context_ids == {1, 2}
    assert 99 not in context_ids


def test_roofline_bag_for_doc_uncompacted_member_context_byte_identical(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, _cfg: AxBenchBpbAdvConfig
) -> None:
    """`feature_vocab=None` -> `allowed_ids=None` threaded through -> the FIXED
    context bag is BYTE-IDENTICAL to the segment's raw, unrestricted ids -- the
    legacy no-op contract, pinned at the REAL caller boundary."""
    model = SimpleNamespace(feature_vocab=None)
    target_bundle = TargetModelBundle(
        model=model, tokenizer=_FakeTokenizer(), max_seq_len=64, target_name="encdec"
    )
    doc = _doc("general_val_ctx_1", [0, 1, 2], [3, 4])
    seg = ContextFeatureSegment(
        feature_ids=(1, 99, 2), buckets=(10, 20, 30), char_start=0, char_end=3
    )
    warm = FeatureConditioning.from_dict({9: 0})
    state = _RooflineGeneralValState.for_budget(
        GeneralValSignalBudget(k_pos=3, k_neg=2, positive_bucket=0, negative_bucket=1)
    )
    state.real_bags[doc.sample_id] = warm
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="general-val-ctx-legacy")
    combo = _combo(method_name="feature_diffmean_roofline_encdec", coefficient_str="None")

    members, calls = _spy_on_member_builder(monkeypatch)

    general_val._roofline_bag_for_doc(
        combo=combo,
        family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        doc=doc,
        segments=(seg,),
        roofline_state=state,
        target_bundle=target_bundle,
        paths=paths,
        tokenizer=_FakeTokenizer(),
        device=t.device("cpu"),
        corpus_signature=_CORPUS,
        cfg=_cfg,
    )

    assert len(members) == 1
    assert calls[0].get("allowed_ids") is None
    context_entries = [c.entries for c in members[0].segment_conditions[:-1]]
    # FeatureConditioning.from_dict sorts entries by feature id (1, 2, 99) --
    # unrestricted (all three ids kept), unchanged from the raw segment.
    assert context_entries == [((1, 10), (2, 30), (99, 20))]
