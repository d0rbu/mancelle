"""Characterization tests for direction-intervention dispatch, context-segment
resolution, and context-cache manifest loading in BPB-Adv scoring.

These cover the branches in ``scoring.py`` not exercised by
``test_scoring_aggregation.py``: the diffmean/linear_probe direction dispatch
of ``_load_concept_direction`` (including the unsupported-baseline and
missing-file errors), the encdec context-signal dispatch of
``_resolve_context_segments`` (fixed_context projection / not_applicable
passthrough / unsupported-mode coerce raise) — the surviving paths the
recomputed_context removal (#506) simplified — and the absent-manifest error of
``load_context_segments_by_prefix_hash``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save, save_file

import steering.eval.axbench_bpb_adv.intervention_build as intervention_build
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.intervention_build import _load_concept_direction
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    _concepts_from_entries,
    _read_concepts,
    _resolve_context_segments,
    load_context_segments_by_prefix_hash,
)
from steering.eval.target_conditioning import ContextFeatureSegment


class _CharTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _write_direction(
    *,
    source_paths: SourcePaths,
    baseline: str,
    target: str,
    mode: str,
    concept_id: int,
) -> Path:
    if baseline == "diffmean":
        path = source_paths.diffmean_direction(target, mode, concept_id)
    else:
        path = source_paths.linear_probe_direction(target, mode, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.ones(4)},
        str(path),
        metadata={
            "concept_id": str(concept_id),
            "layer": "8",
            "mode": mode,
            "target_model": target,
        },
    )
    return path


def _method(*, baseline: str, target: str = "decoder_only") -> Any:
    return SimpleNamespace(
        baseline=baseline,
        target_model=target,
        mode=SimpleNamespace(value=SteeringMode.POSITIVE.value),
    )


# ---------------------------------------------------------------------------
# _load_concept_direction: additive counter-direction loader.
# ---------------------------------------------------------------------------


def test_load_concept_direction_diffmean(tmp_path: Path) -> None:
    source_paths = SourcePaths(tmp_path)
    _write_direction(
        source_paths=source_paths,
        baseline="diffmean",
        target="decoder_only",
        mode=SteeringMode.POSITIVE.value,
        concept_id=1,
    )

    direction, layer, _metadata = _load_concept_direction(
        family="diffmean",
        mode=SteeringMode.POSITIVE.value,
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        source_paths=source_paths,
        expected_layer=8,
    )

    assert layer == 8
    assert t.equal(direction, t.ones(4))


def test_load_concept_direction_linear_probe(tmp_path: Path) -> None:
    source_paths = SourcePaths(tmp_path)
    _write_direction(
        source_paths=source_paths,
        baseline="linear_probe",
        target="decoder_only",
        mode=SteeringMode.POSITIVE.value,
        concept_id=1,
    )

    direction, layer, _metadata = _load_concept_direction(
        family="linear_probe",
        mode=SteeringMode.POSITIVE.value,
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        source_paths=source_paths,
        expected_layer=8,
    )
    assert layer == 8
    assert t.equal(direction, t.ones(4))


def test_load_concept_direction_uses_requested_target(tmp_path: Path) -> None:
    """The counter direction is loaded for the requested target's residual basis:
    an encdec baseline must read baselines/diffmean/encdec/... (NOT decoder_only)."""
    source_paths = SourcePaths(tmp_path)
    # Write ONLY the encdec direction (a distinct value), not the decoder_only one.
    encdec_path = source_paths.diffmean_direction(
        "encdec", SteeringMode.POSITIVE.value, 1
    )
    encdec_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.full((4,), 7.0)},
        str(encdec_path),
        metadata={
            "concept_id": "1",
            "layer": "8",
            "mode": "positive",
            "target_model": "encdec",
        },
    )

    direction, layer, _metadata = _load_concept_direction(
        family="diffmean",
        mode=SteeringMode.POSITIVE.value,
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        source_paths=source_paths,
        expected_layer=8,
        target="encdec",
    )
    assert layer == 8
    assert t.equal(direction, t.full((4,), 7.0))  # the encdec file, not decoder_only


def test_load_concept_direction_rejects_unsupported_family(tmp_path: Path) -> None:
    with pytest.raises(
        AxBenchBpbAdvError, match="supports only diffmean and"
    ):
        _load_concept_direction(
            family="prompting",
            mode=SteeringMode.POSITIVE.value,
            concept=cast(Any, SimpleNamespace(concept_id=1)),
            source_paths=SourcePaths(tmp_path),
            expected_layer=8,
        )


def test_load_concept_direction_rejects_missing_direction_file(tmp_path: Path) -> None:
    # No direction file written -> the path does not exist.
    with pytest.raises(AxBenchBpbAdvError, match="real regular file"):
        _load_concept_direction(
            family="diffmean",
            mode=SteeringMode.POSITIVE.value,
            concept=cast(Any, SimpleNamespace(concept_id=1)),
            source_paths=SourcePaths(tmp_path),
            expected_layer=8,
        )


def test_load_concept_direction_rejects_file_without_direction_tensor(
    tmp_path: Path,
) -> None:
    # GAP B.9: the file EXISTS but carries no "direction" tensor (only some other
    # key) -> raise loudly naming the missing 'direction' tensor, never silently
    # crash on get_tensor("direction"). Pins the in-file key guard
    # (intervention_build ~:95).
    source_paths = SourcePaths(tmp_path)
    path = source_paths.diffmean_direction(
        "decoder_only", SteeringMode.POSITIVE.value, 1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"not_direction": t.ones(4)},
        str(path),
        metadata={
            "concept_id": "1",
            "layer": "8",
            "mode": "positive",
            "target_model": "decoder_only",
        },
    )

    with pytest.raises(AxBenchBpbAdvError, match="tensor key 'direction'"):
        _load_concept_direction(
            family="diffmean",
            mode=SteeringMode.POSITIVE.value,
            concept=cast(Any, SimpleNamespace(concept_id=1)),
            source_paths=source_paths,
            expected_layer=8,
        )


def test_load_concept_direction_layer_metadata_present_and_matching(
    tmp_path: Path,
) -> None:
    # GAP B.10: the mutation-survivable EQUALITY branch -- layer metadata is
    # PRESENT and MATCHES expected_layer, so the guard passes WITHOUT raising and
    # the (direction, expected_layer) is returned. The mismatch case is already
    # covered; this pins the matching arm so inverting `!=` to `==` (which would
    # raise on a correct match) turns RED.
    source_paths = SourcePaths(tmp_path)
    path = source_paths.diffmean_direction(
        "decoder_only", SteeringMode.POSITIVE.value, 1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.ones(4)},
        str(path),
        metadata={
            "concept_id": "1",
            "layer": "8",
            "mode": "positive",
            "target_model": "decoder_only",
        },
    )

    direction, layer, metadata = _load_concept_direction(
        family="diffmean",
        mode=SteeringMode.POSITIVE.value,
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        source_paths=source_paths,
        expected_layer=8,
    )
    assert layer == 8
    assert t.equal(direction, t.ones(4))
    assert metadata == {
        "concept_id": "1",
        "layer": "8",
        "mode": "positive",
        "target_model": "decoder_only",
    }


def test_load_concept_direction_uses_certified_bytes_without_opening_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = SourcePaths(tmp_path)
    path = source_paths.diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE.value,
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.full((768,), -99.0)},
        str(path),
        metadata={
            "concept_id": "1",
            "layer": "8",
            "mode": "positive",
            "target_model": "decoder_only",
        },
    )
    certified = bytes(
        save(
            {"direction": t.linspace(-1.0, 1.0, 768)},
            metadata={
                "concept_id": "1",
                "layer": "8",
                "mode": "positive",
                "target_model": "decoder_only",
            },
        )
    )
    calls: list[dict[str, Any]] = []

    class Validation:
        def direction_artifact_bytes(self, **kwargs: Any) -> bytes:
            calls.append(kwargs)
            return certified

    monkeypatch.setattr(
        "steering.eval.axbench_base.direction_artifacts._read_regular_file_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clustered direction path reopened")
        ),
    )

    direction, layer, _metadata = _load_concept_direction(
        family="diffmean",
        mode="positive",
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        source_paths=source_paths,
        expected_layer=8,
        target="decoder_only",
        source_validation=cast(Any, Validation()),
    )

    assert layer == 8
    assert t.equal(direction, t.linspace(-1.0, 1.0, 768))
    assert not t.equal(direction, t.full((768,), -99.0))
    assert calls == [
        {
            "family": "diffmean",
            "target": "decoder_only",
            "mode": "positive",
            "concept_id": 1,
        }
    ]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"not-safetensors", "not valid safetensors"),
        (
            bytes(
                save(
                    {"other": t.ones(4)},
                    metadata={
                        "concept_id": "1",
                        "layer": "8",
                        "mode": "positive",
                        "target_model": "decoder_only",
                    },
                )
            ),
            "tensor key 'direction'",
        ),
        (
            bytes(
                save(
                    {"direction": t.ones(4)},
                    metadata={
                        "concept_id": "1",
                        "layer": "7",
                        "mode": "positive",
                        "target_model": "decoder_only",
                    },
                )
            ),
            "layer mismatch",
        ),
    ],
)
def test_load_concept_direction_rejects_corrupted_certified_payload(
    tmp_path: Path,
    payload: bytes,
    match: str,
) -> None:
    validation = SimpleNamespace(
        direction_artifact_bytes=lambda **_kwargs: payload,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _load_concept_direction(
            family="diffmean",
            mode="positive",
            concept=cast(Any, SimpleNamespace(concept_id=1)),
            source_paths=SourcePaths(tmp_path),
            expected_layer=8,
            target="decoder_only",
            source_validation=cast(Any, validation),
        )


def test_build_steered_intervention_threads_certified_base_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"certified-base-direction"
    captured: dict[str, Any] = {}
    base = object()

    class Validation:
        def direction_artifact_bytes(self, **kwargs: Any) -> bytes:
            captured["coordinates"] = kwargs
            return payload

    def _build(**kwargs: Any) -> object:
        captured["build"] = kwargs
        return base

    monkeypatch.setattr(intervention_build, "build_intervention", _build)
    monkeypatch.setattr(
        intervention_build,
        "apply_additive_counter",
        lambda intervention, **_kwargs: intervention,
    )
    method = SimpleNamespace(
        name="diffmean_decoder_only",
        baseline=BaselineKind.DIFFMEAN,
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )

    result = intervention_build.build_steered_intervention(
        method=method,
        concept=cast(Any, SimpleNamespace(concept_id=3)),
        coefficient=1.0,
        adversarial_method="none",
        adversarial_strength=0.0,
        source_paths=SourcePaths(tmp_path),
        diffmean_layer=8,
        target="decoder_only",
        counter_direction_family="diffmean",
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=False,
        source_validation=cast(Any, Validation()),
    )

    assert result is base
    assert captured["coordinates"] == {
        "family": "diffmean",
        "target": "decoder_only",
        "mode": "positive",
        "concept_id": 3,
    }
    assert captured["build"]["direction_payload"] is payload


def test_model_backed_scorer_threads_the_coordinate_selected_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.scoring as scoring

    payload = b"certified-direction-payload"
    accessor_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []
    sentinel = object()

    class Validation:
        def direction_artifact_bytes(self, **kwargs: Any) -> bytes:
            accessor_calls.append(kwargs)
            return payload

    def build(**kwargs: Any) -> object:
        build_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(scoring, "build_intervention", build)
    scorer = ModelBackedDualCompletionScorer(
        target_bundles={
            "decoder_only": cast(
                Any,
                SimpleNamespace(model=SimpleNamespace(feature_vocab=None)),
            )
        },
        target_model_metas={"decoder_only": SimpleNamespace()},
        concepts_by_id={1: cast(Any, SimpleNamespace(concept_id=1))},
        source_paths=SourcePaths(tmp_path),
        tokenizer=SimpleNamespace(),
        device=t.device("cpu"),
        diffmean_layer=6,
        inactive_sampler=None,
        apply_max_act_scale=False,
        cluster_contract=cast(Any, object()),
        source_validation=cast(Any, Validation()),
    )
    method = SimpleNamespace(
        name="diffmean_decoder_only",
        baseline=BaselineKind.DIFFMEAN,
        mode=SteeringMode.POSITIVE,
    )

    result = scorer._base_steered_intervention(
        method=method,
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        coefficient=2.0,
        candidate_row={
            "method_name": "diffmean_decoder_only",
            "concept_id": 1,
            "core_prefix_id": "prefix",
            "coefficient_str": "2.0",
            "target_model": "decoder_only",
        },
    )

    assert result is sentinel
    assert accessor_calls == [
        {
            "family": "diffmean",
            "target": "decoder_only",
            "mode": "positive",
            "concept_id": 1,
        }
    ]
    assert len(build_calls) == 1
    assert build_calls[0]["direction_payload"] is payload
    assert build_calls[0]["cluster_contract"] is scorer.cluster_contract


# ---------------------------------------------------------------------------
# _resolve_context_segments: encdec context-signal dispatch. Reached only when
# target_model == "encdec" (decoder_only short-circuits to () in the caller).
# ---------------------------------------------------------------------------


def _seed() -> tuple[
    dict[str, tuple[ContextFeatureSegment, ...]],
    dict[str, Any],
]:
    source_segments = (
        ContextFeatureSegment(
            feature_ids=(3,),
            buckets=(4,),
            char_start=0,
            char_end=2,
        ),
    )
    by_hash = {"deadbeef": source_segments}
    candidate_row: dict[str, Any] = {
        "candidate_id": "c0",
        "prefix_hash": "deadbeef",
        # Source "ab" (tokens [0, 97, 98]); attacked replaces 'b'(98) with 'x'(120),
        # length-preserving so fixed_context projection succeeds.
        "source_prompt_tokens": [0, 97, 98],
        "attacked_prompt_tokens": [0, 97, 120],
        "context_signal_mode": "fixed_context",
    }
    return by_hash, candidate_row


def test_resolve_context_segments_encdec_fixed_context_projects() -> None:
    by_hash, candidate_row = _seed()
    candidate_row["context_signal_mode"] = "fixed_context"

    out = _resolve_context_segments(
        candidate_row=candidate_row,
        context_segments_by_prefix_hash=by_hash,
        tokenizer=_CharTokenizer(),
    )

    assert len(out) == 1
    projected = out[0]
    assert projected.feature_ids == (3,)
    assert projected.buckets == (4,)
    assert (projected.char_start, projected.char_end) == (0, 2)


def test_resolve_context_segments_encdec_not_applicable_passes_through() -> None:
    by_hash, candidate_row = _seed()
    candidate_row["context_signal_mode"] = "not_applicable"
    source_segments = by_hash["deadbeef"]

    out = _resolve_context_segments(
        candidate_row=candidate_row,
        context_segments_by_prefix_hash=by_hash,
        tokenizer=_CharTokenizer(),
    )

    assert out is source_segments


def test_resolve_context_segments_rejects_recomputed_context() -> None:
    # recomputed_context was removed (#506); coercing it now hard-fails at the
    # ContextSignalMode.coerce entry of _resolve_context_segments.
    by_hash, candidate_row = _seed()
    candidate_row["context_signal_mode"] = "recomputed_context"

    with pytest.raises(
        AxBenchBpbAdvError, match="Unknown context signal mode 'recomputed_context'"
    ):
        _resolve_context_segments(
            candidate_row=candidate_row,
            context_segments_by_prefix_hash=by_hash,
            tokenizer=_CharTokenizer(),
        )


# ---------------------------------------------------------------------------
# load_context_segments_by_prefix_hash: absent manifest.
# ---------------------------------------------------------------------------


def test_load_context_segments_raises_when_manifest_absent(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    # The context_cache dir / manifest.json was never created.
    assert not paths.context_cache_manifest().exists()

    with pytest.raises(AxBenchBpbAdvError, match="requires a BPB context cache"):
        load_context_segments_by_prefix_hash(paths)


# ---------------------------------------------------------------------------
# _read_concepts: concepts.json shape guards (scoring.py ~:610/~:616).
# ---------------------------------------------------------------------------


def _write_concepts_json(tmp_path: Path, payload: Any) -> SourcePaths:
    source_paths = SourcePaths(tmp_path)
    source_paths.concepts_json().write_text(json.dumps(payload), encoding="utf-8")
    return source_paths


def test_read_concepts_rejects_non_list(tmp_path: Path) -> None:
    # GAP C.11: a concepts.json that is an OBJECT (not a list) must raise loudly --
    # the top-level type guard (scoring ~:610) -- never be iterated as if it were
    # a list of entries.
    source_paths = _write_concepts_json(tmp_path, {"concept_id": 1})
    with pytest.raises(AxBenchBpbAdvError, match="must contain a list"):
        _read_concepts(source_paths)


def test_read_concepts_rejects_non_mapping_entry(tmp_path: Path) -> None:
    # GAP C.12: a list whose ENTRY is not a mapping (here a list) must raise a
    # clear project error naming the bad type -- NOT an opaque TypeError from
    # Concept(**entry). The "; got list" suffix makes the match non-vacuous.
    source_paths = _write_concepts_json(tmp_path, [["concept_id", 1]])
    with pytest.raises(AxBenchBpbAdvError, match="must be objects .*got list"):
        _read_concepts(source_paths)


def test_concepts_from_entries_exactly_matches_path_reader_and_legacy_shape(
    tmp_path: Path,
) -> None:
    payload = [
        {
            "concept_id": 7,
            "feature_index": 70,
            "layer": 12,
            "sae_source": "first",
            "description": "first description",
            "corpus_count": 700,
            "description_source": "fixture",
        },
        {
            "concept_id": 7,
            "feature_index": 71,
            "layer": 13,
            "sae_source": "replacement",
            "description": "replacement description",
            "corpus_count": 701,
            "description_source": "fixture",
        },
        {
            "concept_id": 8,
            "feature_index": 80,
            "layer": 14,
            "sae_source": "second",
            "description": "second description",
            "corpus_count": 800,
            "description_source": "fixture",
        },
    ]
    source_paths = _write_concepts_json(tmp_path, payload)
    retained_entries = json.loads(
        source_paths.concepts_json().read_text(encoding="utf-8")
    )

    from_path = _read_concepts(source_paths)
    from_retained = _concepts_from_entries(retained_entries)

    expected = {
        7: Concept(
            concept_id=7,
            feature_index=71,
            layer=13,
            sae_source="replacement",
            description="replacement description",
            corpus_count=701,
            description_source="fixture",
        ),
        8: Concept(
            concept_id=8,
            feature_index=80,
            layer=14,
            sae_source="second",
            description="second description",
            corpus_count=800,
            description_source="fixture",
        ),
    }
    assert from_retained == from_path == expected
    assert retained_entries == payload


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"concept_id": 1}, "must contain a list"),
        ([["concept_id", 1]], "must be objects .*got list"),
    ],
)
def test_concepts_from_entries_preserves_shape_errors(
    payload: Any,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match):
        _concepts_from_entries(payload)
