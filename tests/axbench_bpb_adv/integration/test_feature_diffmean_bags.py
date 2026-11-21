from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from safetensors.torch import safe_open

import steering.eval.axbench_bpb_adv.feature_diffmean_bags as module
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, OursGcgConfig
from steering.eval.axbench_bpb_adv.feature_diffmean import PrefixedCompletion
from steering.eval.axbench_bpb_adv.feature_diffmean_bags import (
    active_fair_feature_diffmean_modes,
    active_roofline_feature_diffmean_modes,
    build_feature_diffmean_bags_fair,
    build_feature_diffmean_bags_roofline,
    fair_feature_diffmean_methods_active,
    roofline_feature_diffmean_methods_active,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    _members_for_prefixes as _real_members_for_prefixes,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    read_ours_gcg_bag,
    read_ours_gcg_scored_prefixes,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _bag_content_digest(path: Path) -> str:
    """Hash parsed content, not nondeterministic safetensors header bytes."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        payload = {
            "metadata": dict(handle.metadata() or {}),
            "feature_ids": handle.get_tensor("feature_ids").tolist(),
            "buckets": handle.get_tensor("buckets").tolist(),
        }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def _json_content_digest(path: Path) -> str:
    payload = json.loads(path.read_text())
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def _prefix(idx: int, *, mode: SteeringMode = SteeringMode.POSITIVE) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id=f"p{idx}",
        concept_id=7,
        mode=mode,
        prefix_text=f"prefix {idx}",
        prefix_hash=f"h{idx}",
        prefix_tokens=[idx],
        concept_rich_sample_ids=[f"r{idx}"],
        concept_avoiding_sample_ids=[f"a{idx}"],
        concept_rich_output_tokens=[[10 + idx]],
        concept_avoiding_output_tokens=[[20 + idx]],
    )


def _cfg(
    *,
    base_methods: list[str] | None = None,
    representative_target_methods: list[str] | None = None,
) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="src",
        output_root="out",
        feature_dir="features",
        feature_statistics_path="stats.safetensors",
        base_methods=base_methods,
        representative_target_methods=representative_target_methods or [],
        ours_gcg=OursGcgConfig(
            split_seed=0,
        ),
    )


def test_roofline_feature_diffmean_activation_predicates() -> None:
    cfg = _cfg(base_methods=["feature_diffmean_roofline_encdec"])
    assert active_roofline_feature_diffmean_modes(cfg) == {SteeringMode.POSITIVE}
    assert roofline_feature_diffmean_methods_active(cfg)

    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    assert active_roofline_feature_diffmean_modes(cfg) == set()
    assert not roofline_feature_diffmean_methods_active(cfg)


def test_fair_feature_diffmean_activation_includes_representative_targets() -> None:
    cfg = _cfg(
        base_methods=["diffmean_decoder_only"],
        representative_target_methods=["feature_diffmean_encdec"],
    )
    assert active_fair_feature_diffmean_modes(cfg) == {SteeringMode.POSITIVE}
    assert fair_feature_diffmean_methods_active(cfg)


def test_roofline_feature_diffmean_writes_held_out_per_prefix_bags_and_scored_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefixes = [_prefix(0), _prefix(1), _prefix(2)]
    cfg = _cfg(base_methods=["feature_diffmean_roofline_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline-diffmean")

    prefixed_calls: list[list[str]] = []
    bag_calls: list[tuple[Any, Any, SteeringMode]] = []

    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda _model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda _model, _cfg: (3, 4))
    monkeypatch.setattr(module, "_sae_layer_for_cfg", lambda _cfg: 8)
    monkeypatch.setattr(module, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )
    monkeypatch.setattr(module, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_members_for_prefixes",
        lambda group, **_kwargs: [f"member:{group[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        module,
        "_group_g_steer",
        lambda *_args, **_kwargs: lambda members, bag: len(members) + len(bag),
    )

    def _fake_prefixed(group: list[BpbAttackPrefix], **_kwargs: Any) -> tuple[list[str], list[str]]:
        prefixed_calls.append([prefix.core_prefix_id for prefix in group])
        prefix = group[0]
        return [f"rich:{prefix.core_prefix_id}"], [f"avoid:{prefix.core_prefix_id}"]

    def _fake_bag(rich_items: Any, avoid_items: Any, **kwargs: Any) -> FeatureConditioning:
        bag_calls.append((rich_items, avoid_items, kwargs["mode"]))
        prefix_id = str(rich_items[0]).split(":", 1)[1]
        fid = 100 + int(prefix_id[1:])
        return FeatureConditioning(((fid, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "_prefixed_completions", _fake_prefixed)
    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    build_feature_diffmean_bags_roofline(
        prefixes=prefixes,
        model=SimpleNamespace(feature_vocab=None),
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleNamespace(),
        context_segments_by_prefix_hash={},
    )

    assert prefixed_calls == [["p1"], ["p2"]]
    assert [(rich, avoid, mode) for rich, avoid, mode in bag_calls] == [
        (["rich:p1"], ["avoid:p1"], SteeringMode.POSITIVE),
        (["rich:p2"], ["avoid:p2"], SteeringMode.POSITIVE),
    ]

    bag0_path = paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p1",
    )
    bag1_path = paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p2",
    )
    bag0, meta0 = read_ours_gcg_bag(bag0_path)
    bag1, meta1 = read_ours_gcg_bag(bag1_path)
    assert bag0 == {101: 0}
    assert bag1 == {102: 0}
    assert meta0["mode_tag"] == "feature_diffmean_roofline"
    assert meta0["prefix_id"] == "p1"
    assert meta0["coefficient_str"] == "None"
    assert meta0["scored_split"] == "held_out"
    assert meta1["prefix_id"] == "p2"
    assert "feature_contract_fingerprint" not in meta0
    assert "feature_contract_fingerprint" not in meta1
    assert _bag_content_digest(bag0_path) == (
        "783aa36dd8d2e4991d114662b5093f15430dd11da05d9daf81e14b8bbcbd17bd"
    )
    assert _bag_content_digest(bag1_path) == (
        "aae2dfef6b5d67d73790c99617393599657e0a06f98cb55ff337f2fe17a972f4"
    )
    assert not paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p0",
    ).exists()

    scored_path = paths.ours_gcg_scored_prefixes(
        7,
        SteeringMode.POSITIVE,
        "feature_diffmean_roofline",
    )
    scored = read_ours_gcg_scored_prefixes(scored_path)
    assert scored == ["p1", "p2"]
    assert "feature_contract_fingerprint" not in json.loads(scored_path.read_text())
    assert _json_content_digest(scored_path) == (
        "0bc2e42740add9d446774e2c0e14248d153a964658d4666515e40da251ad4ad0"
    )


def test_fair_feature_diffmean_passes_all_warm_start_items_without_filtering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefixes = [_prefix(0), _prefix(1), _prefix(2)]
    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair-diffmean")

    bag_calls: list[tuple[list[PrefixedCompletion], list[PrefixedCompletion]]] = []

    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda _model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda _model, _cfg: (3, 4))
    monkeypatch.setattr(module, "_sae_layer_for_cfg", lambda _cfg: 8)
    monkeypatch.setattr(module, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:2]), list(group[2:])),
    )
    monkeypatch.setattr(module, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_members_for_prefixes",
        lambda group, **_kwargs: [f"member:{prefix.core_prefix_id}" for prefix in group],
    )
    monkeypatch.setattr(
        module,
        "_group_g_steer",
        lambda *_args, **_kwargs: lambda members, bag: len(members) + len(bag),
    )

    def _fake_prefixed(
        group: list[BpbAttackPrefix], **_kwargs: Any
    ) -> tuple[list[PrefixedCompletion], list[PrefixedCompletion]]:
        assert [prefix.core_prefix_id for prefix in group] == ["p0", "p1"]
        return (
            [
                PrefixedCompletion(prefix_text="3 4 5", completion_text=" 1"),
                PrefixedCompletion(prefix_text="3 4 5", completion_text="1"),
            ],
            [
                PrefixedCompletion(prefix_text="3 4 5", completion_text=" 2"),
                PrefixedCompletion(prefix_text="3 4 5", completion_text="2"),
            ],
        )

    def _fake_bag(
        rich_items: list[PrefixedCompletion],
        avoid_items: list[PrefixedCompletion],
        **kwargs: Any,
    ) -> FeatureConditioning:
        bag_calls.append((rich_items, avoid_items))
        return FeatureConditioning(((100, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "_prefixed_completions", _fake_prefixed)
    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=SimpleNamespace(feature_vocab=None),
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleNamespace(),
        context_segments_by_prefix_hash={},
    )

    assert len(bag_calls) == 1
    rich_items, avoid_items = bag_calls[0]
    assert [item.completion_text for item in rich_items] == [" 1", "1"]
    assert [item.completion_text for item in avoid_items] == [" 2", "2"]

    bag_path = paths.ours_gcg_bag(
        7, SteeringMode.POSITIVE, 0.0, "feature_diffmean"
    )
    _bag, metadata = read_ours_gcg_bag(bag_path)
    assert metadata["n_rich_completions"] == "2"
    assert metadata["n_avoid_completions"] == "2"
    assert "feature_contract_fingerprint" not in metadata
    assert _bag_content_digest(bag_path) == (
        "f1ed0bf810cb6ce85d043ee51a71ee8b73b65baedfa4b3955edc4511d017038d"
    )

    scored_path = paths.ours_gcg_scored_prefixes(
        7,
        SteeringMode.POSITIVE,
        "feature_diffmean",
    )
    scored = read_ours_gcg_scored_prefixes(scored_path)
    assert scored == ["p2"]
    assert "feature_contract_fingerprint" not in json.loads(scored_path.read_text())
    assert _json_content_digest(scored_path) == (
        "233be53cbcbb43137b185d4e14d52afe562de7f925503c094fde83a536ad20d4"
    )


def test_roofline_feature_diffmean_raises_when_warm_start_bag_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefixes = [_prefix(0), _prefix(1)]
    cfg = _cfg(base_methods=["feature_diffmean_roofline_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline-fail-fast")

    bag_calls: list[tuple[list[PrefixedCompletion], list[PrefixedCompletion]]] = []

    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda _model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda _model, _cfg: (3, 4))
    monkeypatch.setattr(module, "_sae_layer_for_cfg", lambda _cfg: 8)
    monkeypatch.setattr(module, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[1:]), list(group[:1])),
    )
    monkeypatch.setattr(module, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_members_for_prefixes",
        lambda group, **_kwargs: [f"member:{group[0].core_prefix_id}"],
    )
    monkeypatch.setattr(
        module,
        "_group_g_steer",
        lambda *_args, **_kwargs: lambda members, bag: len(members) + len(bag),
    )

    def _fake_prefixed(
        group: list[BpbAttackPrefix], **_kwargs: Any
    ) -> tuple[list[PrefixedCompletion], list[PrefixedCompletion]]:
        prefix_id = group[0].core_prefix_id
        if prefix_id == "p0":
            return (
                [PrefixedCompletion(prefix_text="3 4 5", completion_text="1")],
                [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 2")],
            )
        return (
            [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 1")],
            [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 2")],
        )

    def _fake_bag(
        rich_items: list[PrefixedCompletion],
        avoid_items: list[PrefixedCompletion],
        **kwargs: Any,
    ) -> FeatureConditioning:
        bag_calls.append((rich_items, avoid_items))
        if rich_items[0].completion_text == "1":
            raise RuntimeError("bad completion span")
        return FeatureConditioning(((101, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "_prefixed_completions", _fake_prefixed)
    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    with pytest.raises(RuntimeError, match="bad completion span"):
        build_feature_diffmean_bags_roofline(
            prefixes=prefixes,
            model=SimpleNamespace(feature_vocab=None),
            base_model=SimpleNamespace(),
            sae=SimpleNamespace(),
            percentile_table=t.zeros((1, 1)),
            cfg=cfg,
            paths=paths,
            device="cpu",
            tokenizer=SimpleNamespace(),
            context_segments_by_prefix_hash={},
        )

    assert len(bag_calls) == 1
    assert not paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p0",
    ).exists()
    assert not paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p1",
    ).exists()
    assert not paths.ours_gcg_scored_prefixes(
        7,
        SteeringMode.POSITIVE,
        "feature_diffmean_roofline",
    ).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Vocab restriction (issue #695 PR-1 task 2): `allowed_ids` threaded from the
# loaded model's `feature_vocab` into every `feature_diffmean_bag` call.
# ─────────────────────────────────────────────────────────────────────────────


def _common_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "assert_single_positive_bucket", lambda _model: (0, 1))
    monkeypatch.setattr(module, "resolve_k", lambda _model, _cfg: (1, 1))
    monkeypatch.setattr(module, "_sae_layer_for_cfg", lambda _cfg: 8)
    monkeypatch.setattr(module, "encdec_dec_sequence_len", lambda _model: 64)
    monkeypatch.setattr(module, "_assert_prefix_tokens_roundtrip", lambda *_a, **_k: None)
    monkeypatch.setattr(
        module,
        "_members_for_prefixes",
        lambda group, **_kwargs: [f"member:{p.core_prefix_id}" for p in group],
    )
    monkeypatch.setattr(
        module,
        "_group_g_steer",
        lambda *_a, **_k: lambda members, bag: len(members) + len(bag),
    )
    monkeypatch.setattr(
        module,
        "_prefixed_completions",
        lambda group, **_kwargs: (
            [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 1")],
            [PrefixedCompletion(prefix_text="3 4 5", completion_text=" 2")],
        ),
    )


def test_fair_feature_diffmean_threads_allowed_ids_from_model_feature_vocab(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``allowed_ids`` passed to ``feature_diffmean_bag`` is the loaded model's
    OWN ``encoder.feature_vocab_ids`` buffer -- proving the V-restriction
    interface (Task 1's ``feature_vocab_allowed_ids``) is actually wired into
    the fair builder, not just accepted as a dead kwarg."""
    prefixes = [_prefix(0), _prefix(1)]
    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair-vocab")

    allowed_ids = t.tensor([5, 9], dtype=t.int64)
    model = SimpleNamespace(
        # non-None -> "compacted"; carries a fingerprint() because the write side
        # now stamps checkpoint_vocab_fingerprint(model) into the bag (Task 5).
        feature_vocab=SimpleNamespace(fingerprint=lambda: "fp-fair-compacted"),
        encoder=SimpleNamespace(feature_vocab_ids=allowed_ids),
    )

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )

    captured: list[Any] = []

    def _fake_bag(rich_items: Any, avoid_items: Any, **kwargs: Any) -> FeatureConditioning:
        captured.append(kwargs.get("allowed_ids"))
        return FeatureConditioning(((100, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleNamespace(),
        context_segments_by_prefix_hash={},
    )

    assert len(captured) == 1
    assert captured[0] is allowed_ids  # the model's OWN buffer, not a copy


def test_roofline_feature_diffmean_threads_allowed_ids_from_model_feature_vocab(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same threading proof as the fair builder, for the roofline builder --
    including across MULTIPLE per-prefix ``feature_diffmean_bag`` calls."""
    prefixes = [_prefix(0), _prefix(1), _prefix(2)]
    cfg = _cfg(base_methods=["feature_diffmean_roofline_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline-vocab")

    allowed_ids = t.tensor([3, 7, 11], dtype=t.int64)
    model = SimpleNamespace(
        # non-None -> "compacted"; carries a fingerprint() because the write side
        # now stamps checkpoint_vocab_fingerprint(model) into the bag (Task 5).
        feature_vocab=SimpleNamespace(fingerprint=lambda: "fp-roofline-compacted"),
        encoder=SimpleNamespace(feature_vocab_ids=allowed_ids),
    )

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )

    captured: list[Any] = []

    def _fake_bag(rich_items: Any, avoid_items: Any, **kwargs: Any) -> FeatureConditioning:
        captured.append(kwargs.get("allowed_ids"))
        return FeatureConditioning(((100, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    build_feature_diffmean_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleNamespace(),
        context_segments_by_prefix_hash={},
    )

    # 2 held-out prefixes (p1, p2) -> 2 per-prefix feature_diffmean_bag calls,
    # EACH must receive the model's own allowed_ids buffer.
    assert len(captured) == 2
    assert all(ids is allowed_ids for ids in captured)


def test_fair_feature_diffmean_legacy_model_threads_allowed_ids_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An uncompacted model (``feature_vocab=None``) threads ``allowed_ids=None``
    -- the legacy no-op contract, pinned explicitly at the builder level."""
    prefixes = [_prefix(0), _prefix(1)]
    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair-legacy")
    model = SimpleNamespace(feature_vocab=None)

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )

    captured: list[Any] = []

    def _fake_bag(rich_items: Any, avoid_items: Any, **kwargs: Any) -> FeatureConditioning:
        captured.append(kwargs.get("allowed_ids"))
        return FeatureConditioning(((100, kwargs["positive_bucket"]),))

    monkeypatch.setattr(module, "feature_diffmean_bag", _fake_bag)

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleNamespace(),
        context_segments_by_prefix_hash={},
    )

    assert captured == [None]


# ─────────────────────────────────────────────────────────────────────────────
# Context-bag restriction wiring (issue #720 PR-2 Task 2): `allowed_ids` --
# already computed above and threaded into `feature_diffmean_bag` (the WARM/
# selection bag) -- was NEVER passed to `_members_for_prefixes`, so a compacted
# derive built each ClassMarginMember's FIXED context bags from UN-restricted
# (original-SAE-id) context-cache features. `build_class_margin_members` (the
# primitive `_members_for_prefixes` wraps) already restricts correctly when
# given `allowed_ids` -- see
# tests/axbench_bpb_adv/unit/test_gcg_members_vocab.py -- but the caller here
# never passed it. These tests exercise the REAL, un-stubbed
# `_members_for_prefixes` (only leaf SAE/g_steer calls are stubbed) to prove
# the fix at the actual call site, not just the primitive.
# ─────────────────────────────────────────────────────────────────────────────


def _compacted_model(allowed_ids: t.Tensor, *, fingerprint: str) -> Any:
    return SimpleNamespace(
        feature_vocab=SimpleNamespace(fingerprint=lambda: fingerprint),
        encoder=SimpleNamespace(feature_vocab_ids=allowed_ids),
    )


def _member_context_ids(member: Any) -> set[int]:
    """Every feature id across a ClassMarginMember's FIXED context bags.

    ``segment_conditions[:-1]`` on both ``rich`` and ``avoid`` -- the
    generation bag at ``[-1]`` is always empty at build time (the GCG
    optimizer substitutes into it later), so it never contributes ids.
    """
    ids: set[int] = set()
    for leaf in (member.rich, member.avoid):
        for bag in leaf.segment_conditions[:-1]:
            ids.update(fid for fid, _bucket in bag.entries)
    return ids


def test_fair_feature_diffmean_members_for_prefixes_restricts_context_to_v(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The REAL derive-caller path (`build_feature_diffmean_bags_fair` ->
    `_members_for_prefixes` -> `build_class_margin_members`, none stubbed) must
    restrict every built member's FIXED context bags to the compacted model's
    V -- the SAME `allowed_ids` already threaded into `feature_diffmean_bag`.

    Mutation check: deleting the `allowed_ids=allowed_ids` kwarg this task adds
    to the `_members_for_prefixes` calls reproduces today's dead-wiring bug --
    id 999 (outside V) survives into the built members -- and REDs this test.
    """
    prefixes = [_prefix(0), _prefix(1)]
    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair-ctx-vocab")

    allowed_ids = t.tensor([100, 200], dtype=t.int64)
    model = _compacted_model(allowed_ids, fingerprint="fp-fair-ctx")

    # Each segment mixes an in-V id with an OOV id (999) -- NOT fully OOV, so
    # restriction only DROPS 999 rather than raising a fully-emptied-segment
    # error (that raise path is already covered by test_gcg_members_vocab.py).
    context_segments_by_prefix_hash: dict[
        str, tuple[ContextFeatureSegment, ...]
    ] = {
        "h0": (
            ContextFeatureSegment(
                feature_ids=(100, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
        "h1": (
            ContextFeatureSegment(
                feature_ids=(200, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
    }

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )
    monkeypatch.setattr(
        module,
        "feature_diffmean_bag",
        lambda *_a, **kwargs: FeatureConditioning(((100, kwargs["positive_bucket"]),)),
    )

    captured: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> list[Any]:
        members = _real_members_for_prefixes(*args, **kwargs)
        captured.extend(members)
        return members

    # Overrides _common_stubs' dummy `_members_for_prefixes` with a spy that
    # calls through to the REAL implementation.
    monkeypatch.setattr(module, "_members_for_prefixes", _spy)

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleTokenizer(),
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
    )

    assert len(captured) == 2  # 1 optimize (p0) + 1 held-out (p1) member
    all_ids: set[int] = set()
    for member in captured:
        all_ids |= _member_context_ids(member)
    assert all_ids == {100, 200}
    assert 999 not in all_ids


def test_roofline_feature_diffmean_members_for_prefixes_restricts_context_to_v(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same threading proof as the fair builder, for the roofline builder --
    including across MULTIPLE per-prefix `_members_for_prefixes` calls."""
    prefixes = [_prefix(0), _prefix(1), _prefix(2)]
    cfg = _cfg(base_methods=["feature_diffmean_roofline_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roofline-ctx-vocab")

    allowed_ids = t.tensor([100, 200], dtype=t.int64)
    model = _compacted_model(allowed_ids, fingerprint="fp-roofline-ctx")

    context_segments_by_prefix_hash: dict[
        str, tuple[ContextFeatureSegment, ...]
    ] = {
        "h0": (
            ContextFeatureSegment(
                feature_ids=(999,), buckets=(1,), char_start=0, char_end=1
            ),
        ),
        "h1": (
            ContextFeatureSegment(
                feature_ids=(100, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
        "h2": (
            ContextFeatureSegment(
                feature_ids=(200, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
    }

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )
    monkeypatch.setattr(
        module,
        "feature_diffmean_bag",
        lambda *_a, **kwargs: FeatureConditioning(((100, kwargs["positive_bucket"]),)),
    )

    captured: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> list[Any]:
        members = _real_members_for_prefixes(*args, **kwargs)
        captured.extend(members)
        return members

    monkeypatch.setattr(module, "_members_for_prefixes", _spy)

    build_feature_diffmean_bags_roofline(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleTokenizer(),
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
    )

    # 2 held-out prefixes (p1, p2) -> 2 per-prefix `_members_for_prefixes`
    # calls, 1 member each (one rich x one avoid completion per _prefix()).
    assert len(captured) == 2
    all_ids: set[int] = set()
    for member in captured:
        all_ids |= _member_context_ids(member)
    assert all_ids == {100, 200}
    assert 999 not in all_ids


def test_fair_feature_diffmean_legacy_model_members_for_prefixes_context_unrestricted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An uncompacted model (`feature_vocab=None`) threads `allowed_ids=None`
    into `_members_for_prefixes` -- a strict no-op, so the built members'
    context bags carry the RAW, un-restricted context-cache ids unchanged
    (byte-identical to before this task's fix existed)."""
    prefixes = [_prefix(0), _prefix(1)]
    cfg = _cfg(base_methods=["feature_diffmean_encdec"])
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair-ctx-legacy")
    model = SimpleNamespace(feature_vocab=None)

    context_segments_by_prefix_hash: dict[
        str, tuple[ContextFeatureSegment, ...]
    ] = {
        "h0": (
            ContextFeatureSegment(
                feature_ids=(100, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
        "h1": (
            ContextFeatureSegment(
                feature_ids=(200, 999), buckets=(0, 1), char_start=0, char_end=1
            ),
        ),
    }

    _common_stubs(monkeypatch)
    monkeypatch.setattr(
        module,
        "split_prefixes_for_concept",
        lambda group, *, ratio, seed: (list(group[:1]), list(group[1:])),
    )
    monkeypatch.setattr(
        module,
        "feature_diffmean_bag",
        lambda *_a, **kwargs: FeatureConditioning(((100, kwargs["positive_bucket"]),)),
    )

    captured: list[Any] = []

    def _spy(*args: Any, **kwargs: Any) -> list[Any]:
        members = _real_members_for_prefixes(*args, **kwargs)
        captured.extend(members)
        return members

    monkeypatch.setattr(module, "_members_for_prefixes", _spy)

    build_feature_diffmean_bags_fair(
        prefixes=prefixes,
        model=model,
        base_model=SimpleNamespace(),
        sae=SimpleNamespace(),
        percentile_table=t.zeros((1, 1)),
        cfg=cfg,
        paths=paths,
        device="cpu",
        tokenizer=SimpleTokenizer(),
        context_segments_by_prefix_hash=context_segments_by_prefix_hash,
    )

    assert len(captured) == 2
    all_ids: set[int] = set()
    for member in captured:
        all_ids |= _member_context_ids(member)
    assert all_ids == {100, 999, 200}  # unrestricted -- nothing dropped
