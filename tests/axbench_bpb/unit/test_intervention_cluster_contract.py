"""Tests for `cluster_contract` threading through `axbench_bpb.intervention`.

F1a ports `cluster_contract` parameter-threading through
`build_intervention` -> `_build_ours_intervention` / `_build_direction_intervention`,
plus the fail-loud live-ours and precomputed-bag boundaries and the exact
in-memory direction-payload contract for clustered DiffMean/linear-probe.

Every test in this file either:
  (a) proves the `cluster_contract=None` path is byte-identical to omitting
      the kwarg entirely (backward compat -- the plan's invariant U), or
  (b) proves a clustered direction call consumes explicitly supplied certified
      bytes without path I/O or raises before partial work.

Fixture helpers are copied inline rather than imported cross-file, per
project convention (see
tests/axbench_bpb/functional/test_intervention_dispatch_negative_coef.py's
own header comment: "cross-file test imports are flaky").
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch as t
from safetensors.torch import save, save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import (
    _build_direction_intervention,
    _build_ours_intervention,
    build_intervention,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention, SteeringMode
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.cluster_feature_contract import ClusterFeatureContract

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeMethod:
    """Minimal stand-in for axbench_base.method_registry.Method."""

    name: str
    baseline: str
    mode: SteeringMode
    target_model: str
    partition_modifier_name: str | None = None


@dataclass
class _FakeConcept:
    concept_id: int
    feature_index: int = 0


def _make_source_paths(tmp_path: Path) -> SourcePaths:
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    return SourcePaths(source_run_dir=run_dir)


def _write_diffmean_direction(sp: SourcePaths, *, target: str, mode: str, concept_id: int) -> Path:
    path = sp.diffmean_direction(target, mode, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.ones(8, dtype=t.float32)},
        str(path),
        metadata={
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": target,
            "layer": "4",
        },
    )
    return path


def _write_bag(path: Path, bag: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_fids = sorted(bag.keys())
    tensors = {
        "feature_ids": t.tensor(sorted_fids, dtype=t.int64),
        "buckets": t.tensor([bag[k] for k in sorted_fids], dtype=t.int64),
    }
    save_file(tensors, str(path))


class _ClusterContractSentinel:
    """A non-None `cluster_contract` stand-in.

    Every raise this file tests fires on an `is not None` identity check
    alone -- none of F1a's new code dereferences an attribute of
    `cluster_contract` (that only happens in F1b's teacher_force.py /
    runtime_state.py, which read real `ClusterFeatureContract` fields and
    so need the oracle's heavier `SimpleNamespace`-built fixture). A bare
    sentinel exercises the exact branch condition F1a actually added
    without overclaiming coverage of unbuilt F1b/G surfaces.

    It now also carries a fingerprint: with the bag stub replaced by real
    validation (#1008), the bag path reads ``feature_contract_fingerprint`` off the
    contract to hand ``read_ours_gcg_bag`` an expectation.
    """

    feature_contract_fingerprint = "b" * 64


_CLUSTER_CONTRACT_SENTINEL = cast(
    ClusterFeatureContract, _ClusterContractSentinel()
)


# ---------------------------------------------------------------------------
# 1. cluster_contract threading is additive: cluster_contract=None (the
#    default) is byte-identical to omitting the kwarg entirely.
# ---------------------------------------------------------------------------


def test_build_intervention_diffmean_cluster_contract_none_matches_omitted(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    _write_diffmean_direction(sp, target="decoder_only", mode="positive", concept_id=1)

    omitted = build_intervention(
        method=method,
        concept=concept,
        coefficient=2.0,
        source_paths=sp,
        diffmean_layer=4,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
    )
    explicit_none = build_intervention(
        method=method,
        concept=concept,
        coefficient=2.0,
        source_paths=sp,
        diffmean_layer=4,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
        cluster_contract=None,
    )
    assert (
        omitted
        == explicit_none
        == Intervention.from_direction(
            kind="diffmean",
            target_model="decoder_only",
            direction=t.ones(8, dtype=t.float32),
            direction_layer=4,
            coefficient=2.0,
        )
    )


def test_build_intervention_control_baseline_ignores_cluster_contract_kwarg(tmp_path):
    """The control baseline never reads cluster_contract at all -- passing a
    non-None sentinel must not change its (kind, feature_conditions)."""
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="control_encdec",
        baseline="control",
        mode=SteeringMode.POSITIVE,
        target_model="encdec",
    )
    concept = _FakeConcept(concept_id=1)

    without = build_intervention(
        method=method,
        concept=concept,
        coefficient=None,
        source_paths=sp,
        diffmean_layer=4,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
    )
    with_sentinel = build_intervention(
        method=method,
        concept=concept,
        coefficient=None,
        source_paths=sp,
        diffmean_layer=4,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
        cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
    )
    assert without == with_sentinel == Intervention.ours(feature_conditions={})


def test_bag_backed_method_cluster_contract_none_still_loads_normally(tmp_path):
    """Regression: the bag-backed path's `cluster_contract=None` behaviour
    is provably unchanged -- a real bag still loads exactly as before."""
    concept = _FakeConcept(concept_id=1)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="test")
    _write_bag(paths.ours_gcg_bag(1, "positive", 0.0, "fair"), {7: 0})

    result = _build_ours_intervention(
        method=SimpleNamespace(
            name="ours_gcg_encdec",
            mode=SteeringMode.POSITIVE,
            partition_modifier_name="ours_gcg_encdec",
        ),
        concept=concept,
        coefficient=0.0,
        source_paths=SourcePaths(source_run_dir=tmp_path),
        target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
        inactive_sampler=None,
        rng=random.Random(0),
        adv_paths=paths,
        model=SimpleNamespace(feature_vocab=None),
        cluster_contract=None,
    )
    assert result == Intervention.ours(feature_conditions={7: 0})


def test_direction_method_cluster_contract_none_still_loads_normally(tmp_path):
    """Regression: the scalar-direction path's `cluster_contract=None`
    behaviour is provably unchanged."""
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    _write_diffmean_direction(sp, target="decoder_only", mode="positive", concept_id=1)

    result = _build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=2.0,
        source_paths=sp,
        diffmean_layer=4,
        apply_max_act_scale=False,
        cluster_contract=None,
    )
    assert result.direction_coefficient == 2.0


# ---------------------------------------------------------------------------
# 2. The fully-portable "live ours_*" reject (oracle-verbatim, zero
#    cross-track dependency).
# ---------------------------------------------------------------------------


def test_live_ours_reject_fires_for_non_bag_backed_method(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="ours_decoder_only",
        baseline="ours",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
        partition_modifier_name=None,  # -> family is None -> "existing ours_*" branch
    )
    concept = _FakeConcept(concept_id=1)

    with pytest.raises(
        AxBenchBpbError,
        match=r"clustered BPB-Adv does not support live ordinary ours method 'ours_decoder_only'",
    ):
        _build_ours_intervention(
            method=method,
            concept=concept,
            coefficient=1.0,
            source_paths=sp,
            target_model_meta=SimpleNamespace(),
            inactive_sampler=None,
            rng=random.Random(0),
            model=SimpleNamespace(feature_vocab=None),
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        )


def test_live_ours_reject_fires_before_loading_concept_features(tmp_path, monkeypatch):
    """Proves the reject fires before `_load_top_k_concept_fids` is ever
    called -- the "existing ours_*" path's first real I/O step."""
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="ours_decoder_only",
        baseline="ours",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    load_spy = Mock(wraps=intervention_module._load_top_k_concept_fids)
    monkeypatch.setattr(intervention_module, "_load_top_k_concept_fids", load_spy)

    with pytest.raises(AxBenchBpbError, match="does not support live ordinary ours method"):
        _build_ours_intervention(
            method=method,
            concept=concept,
            coefficient=1.0,
            source_paths=sp,
            target_model_meta=SimpleNamespace(),
            inactive_sampler=None,
            rng=random.Random(0),
            model=SimpleNamespace(feature_vocab=None),
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        )
    assert load_spy.call_count == 0


def test_live_ours_reject_does_not_fire_for_bag_backed_method(tmp_path):
    """Sanity: a bag-backed method's cluster_contract path hits the OTHER
    (precomputed-bag) stub raise, never this one -- proves the two rejects
    are mutually exclusive by construction (family is not None short-
    circuits before the "existing ours_*" branch is even reached)."""
    concept = _FakeConcept(concept_id=1)
    with pytest.raises(AxBenchBpbError, match="precomputed bag paths must be provided"):
        _build_ours_intervention(
            method=SimpleNamespace(
                name="ours_gcg_encdec",
                mode=SteeringMode.POSITIVE,
                partition_modifier_name="ours_gcg_encdec",
            ),
            concept=concept,
            coefficient=0.0,
            source_paths=SourcePaths(source_run_dir=tmp_path),
            target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
            inactive_sampler=None,
            rng=random.Random(0),
            adv_paths=None,
            model=SimpleNamespace(feature_vocab=None),
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        )


# ---------------------------------------------------------------------------
# 3. The precomputed-bag stub is GONE (#1008): clustered bag consumption now
#    validates the artifact's own feature-contract-fingerprint stamp instead of
#    refusing outright. The prerequisite it waited on
#    (FEATURE_CONTRACT_FINGERPRINT_META_KEY) had already landed.
#
#    The expectation is threaded into read_ours_gcg_bag, which owns the two-way
#    rule: with an expectation the stamp must match; with none, a cluster-stamped
#    artifact is rejected as an unclustered read. Only one artifact is consumed
#    here, so unlike the derive flow there is no second read to bind it to (#938).
# ---------------------------------------------------------------------------


def test_clustered_bag_method_reaches_real_bag_io(tmp_path):
    """A clustered bag-backed method now executes instead of being refused.

    With no adv_paths the failure is the genuine missing-paths error, which is the
    observable difference between a placeholder and working code. This also re-proves
    what the removed stub tests used to prove incidentally: a bag-backed method never
    takes the live-ours reject branch, because `family is not None` short-circuits
    first.
    """
    concept = _FakeConcept(concept_id=1)
    with pytest.raises(AxBenchBpbError, match="precomputed bag paths must be provided"):
        _build_ours_intervention(
            method=SimpleNamespace(
                name="ours_gcg_encdec",
                mode=SteeringMode.POSITIVE,
                partition_modifier_name="ours_gcg_encdec",
            ),
            concept=concept,
            coefficient=0.0,
            source_paths=SourcePaths(source_run_dir=tmp_path),
            target_model_meta=SimpleNamespace(negative_selection=None, num_buckets=2),
            inactive_sampler=None,
            rng=random.Random(0),
            adv_paths=None,
            model=SimpleNamespace(feature_vocab=None),
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        )


# ---------------------------------------------------------------------------
# 4. Clustered diffmean/linear-probe exact-payload handoff.
# ---------------------------------------------------------------------------


def _direction_payload(
    *,
    direction: t.Tensor,
    concept_id: int,
    mode: str,
    target: str,
    layer: int,
    max_activation: float | None = 3.0,
    family: str = "diffmean",
) -> bytes:
    metadata = {
        "concept_id": str(concept_id),
        "mode": mode,
        "target_model": target,
        "layer": str(layer),
        "artifact_kind": f"{family}_direction_v1",
        "feature_contract_fingerprint": "b" * 64,
    }
    if max_activation is not None:
        metadata[MAX_ACTIVATION_METADATA_KEY] = encode_max_activation(max_activation)
    return bytes(
        save(
            {"direction": direction},
            metadata=metadata,
        )
    )


def test_clustered_direction_requires_an_exact_payload(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    with pytest.raises(
        AxBenchBpbError,
        match="requires the exact direction bytes retained by source validation",
    ):
        _build_direction_intervention(
            method=method,
            concept=concept,
            coefficient=1.0,
            source_paths=sp,
            diffmean_layer=4,
            apply_max_act_scale=False,
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        )


def test_unclustered_direction_rejects_missing_path_backed_file(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="diffmean direction file missing"):
        _build_direction_intervention(
            method=_FakeMethod(
                name="diffmean_decoder_only",
                baseline="diffmean",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
            ),
            concept=_FakeConcept(concept_id=1),
            coefficient=1.0,
            source_paths=_make_source_paths(tmp_path),
            diffmean_layer=4,
            apply_max_act_scale=False,
            cluster_contract=None,
        )


def test_direction_payload_rejected_for_non_direction_method(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="valid only for diffmean/linear_probe"):
        build_intervention(
            method=_FakeMethod(
                name="control_decoder_only",
                baseline="control",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
            ),
            concept=_FakeConcept(concept_id=1),
            coefficient=None,
            source_paths=_make_source_paths(tmp_path),
            diffmean_layer=4,
            target_model_meta=SimpleNamespace(),
            inactive_sampler=None,
            tokenizer_nanochat=SimpleNamespace(),
            rng=random.Random(0),
            apply_max_act_scale=False,
            direction_payload=b"not-valid-for-control",
        )


@pytest.mark.parametrize(
    ("baseline", "mode", "coefficient", "expected_coefficient"),
    [
        ("diffmean", SteeringMode.POSITIVE, 1.5, 1.5),
        ("linear_probe", SteeringMode.NEGATIVE, 1.5, -1.5),
    ],
)
def test_clustered_direction_builds_from_payload_without_path_io(
    tmp_path,
    monkeypatch,
    baseline,
    mode,
    coefficient,
    expected_coefficient,
):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name=f"{baseline}_decoder_only",
        baseline=baseline,
        mode=mode,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=2)
    direction = t.arange(8, dtype=t.float32)
    payload = _direction_payload(
        direction=direction,
        concept_id=2,
        mode=mode.value,
        target="decoder_only",
        layer=4,
        family=baseline,
    )
    real_load = intervention_module.load_direction_artifact
    retained_sources: list[bytes] = []

    def load_spy(source, **kwargs):
        assert type(source) is bytes, "clustered direction must use retained bytes"
        retained_sources.append(source)
        return real_load(source, **kwargs)

    monkeypatch.setattr(intervention_module, "load_direction_artifact", load_spy)

    result = _build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=coefficient,
        source_paths=sp,
        diffmean_layer=4,
        apply_max_act_scale=False,
        cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        direction_payload=payload,
    )

    assert result.direction is not None
    assert t.equal(result.direction, direction)
    assert result.direction_layer == 4
    assert result.direction_coefficient == expected_coefficient
    assert retained_sources == [payload]


def test_clustered_direction_scaled_from_payload_metadata(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    result = _build_direction_intervention(
        method=method,
        concept=_FakeConcept(concept_id=2),
        coefficient=1.5,
        source_paths=sp,
        diffmean_layer=4,
        apply_max_act_scale=True,
        cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        direction_payload=_direction_payload(
            direction=t.tensor([6.0, 8.0]),
            concept_id=2,
            mode="positive",
            target="decoder_only",
            layer=4,
            max_activation=7.0,
        ),
    )
    assert result.direction_coefficient == 1.5 * 7.0
    assert t.allclose(cast(t.Tensor, result.direction), t.tensor([0.6, 0.8]))


def test_clustered_direction_scaled_missing_field_raises(tmp_path):
    with pytest.raises(AxBenchBpbError, match="missing 'max_activation'"):
        _build_direction_intervention(
            method=_FakeMethod(
                name="diffmean_decoder_only",
                baseline="diffmean",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
            ),
            concept=_FakeConcept(concept_id=2),
            coefficient=1.0,
            source_paths=_make_source_paths(tmp_path),
            diffmean_layer=4,
            apply_max_act_scale=True,
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
            direction_payload=_direction_payload(
                direction=t.tensor([1.0, 0.0]),
                concept_id=2,
                mode="positive",
                target="decoder_only",
                layer=4,
                max_activation=None,
            ),
        )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("concept_id", "999"),
        ("mode", "negative"),
        ("target_model", "encdec"),
        ("layer", "5"),
    ],
)
def test_clustered_direction_rejects_payload_coordinate_mismatch(
    tmp_path,
    field,
    wrong,
):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    metadata = {
        "concept_id": "1",
        "mode": "positive",
        "target_model": "decoder_only",
        "layer": "4",
        MAX_ACTIVATION_METADATA_KEY: encode_max_activation(3.0),
    }
    metadata[field] = wrong
    payload = bytes(save({"direction": t.ones(8)}, metadata=metadata))
    with pytest.raises(
        AxBenchBpbError,
        match="mismatch",
    ):
        _build_direction_intervention(
            method=method,
            concept=concept,
            coefficient=1.0,
            source_paths=sp,
            diffmean_layer=4,
            apply_max_act_scale=True,
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
            direction_payload=payload,
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not-safetensors", "not valid safetensors"),
        (
            bytes(
                save(
                    {"other": t.ones(8)},
                    metadata={
                        "concept_id": "1",
                        "mode": "positive",
                        "target_model": "decoder_only",
                        "layer": "4",
                        MAX_ACTIVATION_METADATA_KEY: encode_max_activation(3.0),
                    },
                )
            ),
            "tensor key 'direction'",
        ),
        (
            _direction_payload(
                direction=t.tensor([1.0, float("nan")]),
                concept_id=1,
                mode="positive",
                target="decoder_only",
                layer=4,
            ),
            "NaN or Inf",
        ),
        (
            _direction_payload(
                direction=t.ones(8, dtype=t.int64),
                concept_id=1,
                mode="positive",
                target="decoder_only",
                layer=4,
            ),
            "floating point",
        ),
        (
            _direction_payload(
                direction=t.ones(2, 4),
                concept_id=1,
                mode="positive",
                target="decoder_only",
                layer=4,
            ),
            "non-empty 1D",
        ),
        (
            _direction_payload(
                direction=t.empty(0),
                concept_id=1,
                mode="positive",
                target="decoder_only",
                layer=4,
            ),
            "non-empty 1D",
        ),
    ],
)
def test_clustered_direction_rejects_malformed_or_incomplete_payload(
    tmp_path,
    payload,
    match,
):
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    with pytest.raises(AxBenchBpbError, match=match):
        _build_direction_intervention(
            method=method,
            concept=_FakeConcept(concept_id=1),
            coefficient=1.0,
            source_paths=_make_source_paths(tmp_path),
            diffmean_layer=4,
            apply_max_act_scale=False,
            cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
            direction_payload=payload,
        )


def test_unclustered_direction_accepts_source_validated_retained_payload(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    payload = _direction_payload(
        direction=t.ones(8),
        concept_id=1,
        mode="positive",
        target="decoder_only",
        layer=4,
    )
    intervention = _build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=1.0,
        source_paths=sp,
        diffmean_layer=4,
        apply_max_act_scale=False,
        direction_payload=payload,
    )
    assert intervention.direction is not None
    assert t.equal(intervention.direction, t.ones(8))


def test_clustered_direction_payload_reachable_via_build_intervention(tmp_path):
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only",
        baseline="diffmean",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=1)
    direction = t.arange(8, dtype=t.float32)
    result = build_intervention(
        method=method,
        concept=concept,
        coefficient=1.0,
        source_paths=sp,
        diffmean_layer=4,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
        cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        direction_payload=_direction_payload(
            direction=direction,
            concept_id=1,
            mode="positive",
            target="decoder_only",
            layer=4,
        ),
    )

    assert result.direction is not None
    assert t.equal(result.direction, direction)


@pytest.mark.parametrize("width", [768, 1024], ids=lambda value: f"hidden-{value}")
@pytest.mark.parametrize("baseline", ["diffmean", "linear_probe"])
@pytest.mark.parametrize(
    ("mode", "coefficient"),
    [(SteeringMode.POSITIVE, 2.75), (SteeringMode.NEGATIVE, 2.75)],
)
def test_clustered_payload_is_exactly_equal_to_legacy_path_output_at_model_width(
    tmp_path,
    width,
    baseline,
    mode,
    coefficient,
):
    """The new byte handoff changes ownership only, never intervention values."""

    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name=f"{baseline}_decoder_only",
        baseline=baseline,
        mode=mode,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=17)
    direction = t.linspace(-3.0, 5.0, width, dtype=t.float32)
    payload = _direction_payload(
        direction=direction,
        concept_id=17,
        mode=mode.value,
        target="decoder_only",
        layer=11,
        family=baseline,
    )
    path = (
        sp.diffmean_direction("decoder_only", mode.value, 17)
        if baseline == "diffmean"
        else sp.linear_probe_direction("decoder_only", mode.value, 17)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    legacy = _build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=coefficient,
        source_paths=sp,
        diffmean_layer=11,
        apply_max_act_scale=False,
    )
    clustered = _build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=coefficient,
        source_paths=sp,
        diffmean_layer=11,
        apply_max_act_scale=False,
        cluster_contract=_CLUSTER_CONTRACT_SENTINEL,
        direction_payload=payload,
    )

    assert legacy.kind == clustered.kind
    assert legacy.target_model == clustered.target_model
    assert legacy.direction_layer == clustered.direction_layer
    assert legacy.direction_coefficient == clustered.direction_coefficient
    assert legacy.direction is not None
    assert clustered.direction is not None
    assert t.equal(legacy.direction, clustered.direction)
    assert t.equal(clustered.direction, direction)
