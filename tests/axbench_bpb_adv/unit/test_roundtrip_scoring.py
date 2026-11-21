from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
import torch as t

import steering.eval.axbench_base.generation_helpers as generation_helpers
import steering.eval.axbench_bpb_adv.roundtrip_scoring as roundtrip_scoring
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.config import GenerationConfig
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.types import Concept, TargetModelMetadata
from steering.eval.axbench_bpb.types import Intervention, InterventionKind
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    write_evaluation_population,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.roundtrip_scoring import (
    RoundtripGeneratedCompletion,
    RoundtripGenerationInputs,
    _build_model_generate_batch_fn,
    _control_generation_identity_key,
    _float64_bits,
    _resolve_candidate_side_generations,
    _roundtrip_batch_seed,
    _roundtrip_context_attacked_char_start,
    _roundtrip_generation_batch_identity_key,
    _roundtrip_generation_identity_key,
    _roundtrip_generation_inputs,
    _roundtrip_probe_prompt_text,
    _validate_roundtrip_context_text_alignment,
    _validate_roundtrip_probe_matches_concepts,
    _validate_roundtrip_stats_match_probe,
    build_roundtrip_generation_rows,
    run_model_backed_roundtrip_score_stage,
    score_roundtrip_generation_rows,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)
from steering.eval.probe_model import ProbeModel
from steering.eval.roundtrip_steering import (
    RoundtripFeatureScores,
    RoundtripProbeConfig,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SingleEntryContextProvider,
)
from steering.sae_resolver import L0Tier


def _roundtrip_seed(*, identity_key: tuple[Any, ...], master_seed: int) -> int:
    """Reproduce the retired per-sample seed for legacy failure fixtures."""
    payload = json.dumps(
        roundtrip_scoring._jsonable(identity_key),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{int(master_seed)}|{payload}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)


class _TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)

    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        input_ids = self.encode(text)
        out: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [
                (index, index + 1) for index in range(len(input_ids))
            ]
        return out


class _VisibleBosTokenizer(_TinyTokenizer):
    bos_token_id = 0

    def decode(self, ids: Sequence[int]) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token = int(token_id)
            pieces.append("<BOS>" if token == self.bos_token_id else chr(token))
        return "".join(pieces)


class _PairExtractionTokenizer(_TinyTokenizer):
    def encode(self, text: str) -> list[int]:
        return list(range((len(text) + 1) // 2))

    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        input_ids = self.encode(text)
        out: dict[str, Any] = {"input_ids": input_ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [
                (index * 2, min(index * 2 + 2, len(text)))
                for index in range(len(input_ids))
            ]
        return out


class _BoundaryMergeTokenizer:
    """Tokenizer where output-token offsets depend on the source prefix token."""

    bos_token_id = 0
    _decoded = {
        (0,): "<BOS>",
        (0, 2): "<BOS>LONG",
        (0, 2, 3): "<BOS>LONGc",
        (1,): "a",
        (1, 0): "a",
        (1, 0, 2): "ab",
        (1, 0, 2, 3): "abc",
        (2,): "LONG",
        (2, 3): "LONGc",
        (2, 3, 4): "LONGcd",
        (1, 2): "ab",
        (1, 2, 3): "abc",
        (1, 2, 3, 4): "abcd",
    }

    def decode(self, ids: Sequence[int]) -> str:
        return self._decoded[tuple(int(token_id) for token_id in ids)]


_SOURCE = [97, 98]
_RICH = [99, 100]
_AVOID = [101, 102]


def _concepts() -> dict[int, Concept]:
    return {
        1: Concept(
            concept_id=1,
            feature_index=42,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description="feature",
            corpus_count=7,
            description_source="fixture",
        )
    }


def _candidate(
    *,
    candidate_id: str = "c0",
    attack_row_id: str = "r0",
    mode: str = "positive",
    target_model: str = "decoder_only",
    source: list[int] | None = None,
    attacked: list[int] | None = None,
    adversarial_strength: float = 0.0,
    strength_index: int = 0,
    context_signal_mode: str = "fixed_context",
) -> dict[str, Any]:
    source_tokens = list(_SOURCE if source is None else source)
    attacked_tokens = list(source_tokens if attacked is None else attacked)
    editable_start = len(source_tokens)
    editable_count = 0
    replacement_tokens: list[int] = []
    if attacked_tokens != source_tokens:
        editable_start = len(source_tokens) - 1
        editable_count = 1
        replacement_tokens = attacked_tokens[editable_start:]
    return {
        "candidate_id": candidate_id,
        "attack_row_id": attack_row_id,
        "core_prefix_id": "core0",
        "concept_rich_sample_id": "rich0",
        "concept_avoiding_sample_id": "avoid0",
        "concept_id": 1,
        "mode": mode,
        "method_name": "diffmean_decoder_only",
        "target_model": target_model,
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
        "prefix_text": _TinyTokenizer().decode(source_tokens),
        "prefix_hash": "hash0",
        "source_prompt_tokens": source_tokens,
        "source_prompt_token_digest": token_sequence_digest(source_tokens),
        "attacked_prompt_text": _TinyTokenizer().decode(attacked_tokens),
        "attacked_prompt_tokens": attacked_tokens,
        "attacked_prompt_token_digest": token_sequence_digest(attacked_tokens),
        "concept_rich_output_tokens": list(_RICH),
        "concept_avoiding_output_tokens": list(_AVOID),
        "concept_rich_output_digest": token_sequence_digest(_RICH),
        "concept_avoiding_output_digest": token_sequence_digest(_AVOID),
        "editable_token_start": editable_start,
        "editable_token_count": editable_count,
        "replacement_tokens": replacement_tokens,
        "replacement_text": _TinyTokenizer().decode(replacement_tokens),
        "objective_value": None,
        "candidate_metadata_json": "{}",
    }


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _build_intervention(_row: Mapping[str, Any]) -> Intervention:
    return Intervention.none()


def _target_meta(*, segment_size: int = 256) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=None,
        n_sae_features=None,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=segment_size,
    )


def _generation_row_for_scoring(
    *,
    generated_text: str = " generated",
    control_generated_text: str = " control",
) -> dict[str, Any]:
    return {
        **_candidate(),
        "roundtrip_prompt_tokens": [97, 98],
        "roundtrip_prompt_token_digest": token_sequence_digest([97, 98]),
        "roundtrip_prompt_text": "ab",
        "control_roundtrip_prompt_tokens": [97, 98],
        "control_roundtrip_prompt_token_digest": token_sequence_digest([97, 98]),
        "control_roundtrip_prompt_text": "ab",
        "loss_concept_rich": 1.0,
        "loss_concept_avoiding": 3.0,
        "sample_index": 0,
        "reference_segment_token_count": 3,
        "generation_seed": 123,
        "generated_text": generated_text,
        "n_generated_tokens": 0 if generated_text == "" else 3,
        "stop_reason": "bos_token" if generated_text == "" else "max_tokens",
        "control_generation_seed": 124,
        "control_generated_text": control_generated_text,
        "control_n_generated_tokens": 0 if control_generated_text == "" else 2,
        "control_stop_reason": "bos_token" if control_generated_text == "" else "eos",
        "is_control": True,
        "target_feature_id": 42,
    }


def _generate_text(text: str = " generated"):
    calls: list[tuple[str, int, int, list[int], int]] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        assert intervention.kind == "none"
        results: list[RoundtripGeneratedCompletion] = []
        for sample_index, seed in sample_seeds:
            calls.append(
                (
                    str(candidate_row["candidate_id"]),
                    int(sample_index),
                    int(seed),
                    list(inputs.prompt_tokens),
                    int(generation_cfg.max_new_tokens),
                )
            )
            results.append(
                RoundtripGeneratedCompletion(
                    text=text,
                    token_count=3,
                    stop_reason="max_tokens",
                )
            )
        return results

    return _generate, calls


def _generation_rows_guard_kwargs(
    candidates: pa.Table,
    *,
    samples_per_candidate: int = 1,
) -> dict[str, Any]:
    generate, _calls = _generate_text()
    return {
        "candidates": candidates,
        "context_segments_by_prefix_hash": {},
        "tokenizer": _TinyTokenizer(),
        "extraction_tokenizer": _TinyTokenizer(),
        "concepts_by_id": _concepts(),
        "generation_cfg": GenerationConfig(max_new_tokens=5),
        "reference_segment_size_by_target": {"decoder_only": 256},
        "samples_per_candidate": samples_per_candidate,
        "master_seed": 123,
        "baseline_losses_by_candidate_id": {"c0": (1.0, 3.0)},
        "build_intervention_for_row": _build_intervention,
        "build_control_intervention_for_row": _build_intervention,
        "generate_completions": generate,
    }


def test_generation_rows_reject_wrong_candidates_schema() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="expected candidates with CANDIDATES_SCHEMA",
    ):
        build_roundtrip_generation_rows(
            **_generation_rows_guard_kwargs(pa.table({"candidate_id": ["c0"]}))
        )


def test_generation_rows_reject_zero_samples_per_candidate() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="samples_per_candidate must be >= 1",
    ):
        build_roundtrip_generation_rows(
            **_generation_rows_guard_kwargs(
                _table([_candidate()]),
                samples_per_candidate=0,
            )
        )


def test_generation_rows_reject_duplicate_candidate_id() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="duplicate candidate_id='c0'",
    ):
        build_roundtrip_generation_rows(
            **_generation_rows_guard_kwargs(
                _table(
                    [
                        _candidate(candidate_id="c0", attack_row_id="r0"),
                        _candidate(candidate_id="c0", attack_row_id="r1"),
                    ]
                )
            )
        )


def test_model_generate_batch_rejects_target_without_loaded_bundle() -> None:
    generate = _build_model_generate_batch_fn(
        target_bundles={},
        device="cpu",
        lora=None,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="no target bundle loaded for target_model='missing'",
    ):
        generate(
            candidate_row={"target_model": "missing"},
            intervention=Intervention.none(),
            inputs=RoundtripGenerationInputs([97], "a", None),
            generation_cfg=GenerationConfig(max_new_tokens=1),
            sample_seeds=[(0, 1)],
        )


def test_roundtrip_identity_excludes_row_ids() -> None:
    gen_cfg = GenerationConfig(max_new_tokens=5, temperature=0.7, top_k_sampling=10)
    base = _candidate(candidate_id="c0", attack_row_id="r0")
    twin = _candidate(candidate_id="c1", attack_row_id="r1")

    assert _roundtrip_generation_identity_key(
        base,
        prompt_tokens=[*base["attacked_prompt_tokens"]],
        sample_index=0,
        generation_cfg=gen_cfg,
    ) == _roundtrip_generation_identity_key(
        twin,
        prompt_tokens=[*twin["attacked_prompt_tokens"]],
        sample_index=0,
        generation_cfg=gen_cfg,
    )


def test_roundtrip_identity_changes_on_prompt_sample_or_generation_params() -> None:
    row = _candidate()
    gen_cfg = GenerationConfig(max_new_tokens=5, temperature=0.7, top_k_sampling=10)
    base = _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 98],
        sample_index=0,
        generation_cfg=gen_cfg,
    )
    assert base != _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 99],
        sample_index=0,
        generation_cfg=gen_cfg,
    )
    assert base != _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 98],
        sample_index=1,
        generation_cfg=gen_cfg,
    )
    assert base != _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 98],
        sample_index=0,
        generation_cfg=GenerationConfig(
            max_new_tokens=6,
            temperature=0.7,
            top_k_sampling=10,
        ),
    )
    assert base != _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 98],
        sample_index=0,
        generation_cfg=GenerationConfig(
            max_new_tokens=5,
            temperature=0.8,
            top_k_sampling=10,
        ),
    )
    assert base != _roundtrip_generation_identity_key(
        row,
        prompt_tokens=[97, 98],
        sample_index=0,
        generation_cfg=GenerationConfig(
            max_new_tokens=5,
            temperature=0.7,
            top_k_sampling=11,
        ),
    )


def test_control_identity_seed_is_semantic_and_sample_specific() -> None:
    row = _candidate(candidate_id="c0", attack_row_id="r0")
    twin = _candidate(candidate_id="c1", attack_row_id="r1")
    twin["coefficient_str"] = "2.0"
    intervention = Intervention.none()
    inputs = RoundtripGenerationInputs([97, 98], "ab", None)
    generation_cfg = GenerationConfig(
        max_new_tokens=5,
        temperature=0.7,
        top_k_sampling=10,
    )
    first = _control_generation_identity_key(
        row,
        control_intervention=intervention,
        control_inputs=inputs,
        sample_index=0,
        generation_cfg=generation_cfg,
    )
    twin_first = _control_generation_identity_key(
        twin,
        control_intervention=intervention,
        control_inputs=inputs,
        sample_index=0,
        generation_cfg=generation_cfg,
    )
    second_sample = _control_generation_identity_key(
        twin,
        control_intervention=intervention,
        control_inputs=inputs,
        sample_index=1,
        generation_cfg=generation_cfg,
    )

    assert first == twin_first
    assert _roundtrip_seed(identity_key=first, master_seed=123) == _roundtrip_seed(
        identity_key=twin_first,
        master_seed=123,
    )
    assert _roundtrip_seed(identity_key=first, master_seed=123) != _roundtrip_seed(
        identity_key=second_sample,
        master_seed=123,
    )


def test_control_identity_marks_noop_and_uses_bit_exact_counter_provenance() -> None:
    noop_row = _candidate(adversarial_strength=0.0)
    active_row = _candidate(adversarial_strength=1.0)
    active_row["adversarial_method"] = "additive_direction"
    inputs = RoundtripGenerationInputs([97, 98], "ab", None)
    generation_cfg = GenerationConfig(max_new_tokens=5)
    noop = _control_generation_identity_key(
        noop_row,
        control_intervention=Intervention.none(),
        control_inputs=inputs,
        sample_index=0,
        generation_cfg=generation_cfg,
    )
    active_intervention = Intervention.none().with_counter_direction(
        direction=t.tensor([1.0]),
        layer=8,
        coefficient=-1.0,
    )
    active = _control_generation_identity_key(
        active_row,
        control_intervention=active_intervention,
        control_inputs=inputs,
        sample_index=0,
        generation_cfg=generation_cfg,
    )

    assert noop[-1] == ("no_counter",)
    assert active[-1][0] == "additive_direction"
    assert active[-1][2:] == ("positive", 1)
    assert noop != active

    adjacent_strength_row = dict(active_row)
    adjacent_strength_row["adversarial_strength"] = math.nextafter(1.0, 2.0)
    adjacent_strength = _control_generation_identity_key(
        adjacent_strength_row,
        control_intervention=active_intervention,
        control_inputs=inputs,
        sample_index=0,
        generation_cfg=generation_cfg,
    )
    assert active[-1][1] != adjacent_strength[-1][1]
    assert active != adjacent_strength


def test_float64_bits_normalizes_signed_zero_only() -> None:
    assert _float64_bits(-0.0) == _float64_bits(0.0)
    assert _float64_bits(1.0) != _float64_bits(math.nextafter(1.0, 2.0))


def test_prompting_control_identity_uses_control_prompt_tokens() -> None:
    row = _candidate()
    steered_inputs = _roundtrip_generation_inputs(
        candidate_row=row,
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        intervention=Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[80],
        ),
    )
    control_intervention = Intervention.none()
    control_inputs = _roundtrip_generation_inputs(
        candidate_row=row,
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        intervention=control_intervention,
    )
    key = _control_generation_identity_key(
        row,
        control_intervention=control_intervention,
        control_inputs=control_inputs,
        sample_index=0,
        generation_cfg=GenerationConfig(max_new_tokens=5),
    )

    assert control_intervention.prompt_prefix_tokens is None
    assert key[1] == tuple(control_inputs.prompt_tokens) == (97, 98)
    assert key[1] != tuple(steered_inputs.prompt_tokens)


def test_roundtrip_probe_prompt_text_strips_bos_then_decodes_whole_prompt() -> None:
    assert (
        _roundtrip_probe_prompt_text(
            attacked_tokens=[0, 2, 3],
            prompt_prefix_tokens=[1],
            tokenizer=_BoundaryMergeTokenizer(),
        )
        == "abc"
    )
    assert (
        _roundtrip_probe_prompt_text(
            attacked_tokens=[0, 2, 3],
            prompt_prefix_tokens=[],
            tokenizer=_BoundaryMergeTokenizer(),
        )
        == "LONGc"
    )


def test_roundtrip_probe_prompt_text_does_not_strip_nonleading_bos_tokens() -> None:
    assert (
        _roundtrip_probe_prompt_text(
            attacked_tokens=[97, 0, 98],
            prompt_prefix_tokens=[80],
            tokenizer=_VisibleBosTokenizer(),
        )
        == "Pa<BOS>b"
    )


def test_roundtrip_context_attacked_char_start_uses_full_prompt_offsets() -> None:
    assert (
        _roundtrip_context_attacked_char_start(
            prompt_tokens=[80, 0, 97, 98],
            prompt_prefix_token_count=1,
            tokenizer=_VisibleBosTokenizer(),
        )
        == len("P")
    )
    assert (
        _roundtrip_context_attacked_char_start(
            prompt_tokens=[1, 0, 2, 3],
            prompt_prefix_token_count=1,
            tokenizer=_BoundaryMergeTokenizer(),
        )
        == len("a")
    )


def test_validate_roundtrip_context_text_alignment_accepts_stable_suffix() -> None:
    tokenizer = _VisibleBosTokenizer()
    context_text = str(tokenizer.decode([80, 0, 97, 98]))

    _validate_roundtrip_context_text_alignment(
        attacked_tokens=[0, 97, 98],
        context_text=context_text,
        attacked_char_start=len("P"),
        tokenizer=tokenizer,
    )


def test_validate_roundtrip_context_text_alignment_rejects_boundary_merge() -> None:
    tokenizer = _BoundaryMergeTokenizer()
    context_text = str(tokenizer.decode([1, 0, 2, 3]))

    with pytest.raises(AxBenchBpbAdvError, match="cannot align"):
        _validate_roundtrip_context_text_alignment(
            attacked_tokens=[0, 2, 3],
            context_text=context_text,
            attacked_char_start=len("a"),
            tokenizer=tokenizer,
        )


def test_roundtrip_probe_identity_must_match_candidate_concepts() -> None:
    _validate_roundtrip_probe_matches_concepts(
        candidate_rows=[_candidate()],
        concepts_by_id=_concepts(),
        probe_config=RoundtripProbeConfig(layer=8),
    )

    with pytest.raises(AxBenchBpbAdvError, match="roundtrip probe SAE"):
        _validate_roundtrip_probe_matches_concepts(
            candidate_rows=[_candidate()],
            concepts_by_id=_concepts(),
            probe_config=RoundtripProbeConfig(layer=9),
        )


def _concept_with_sae_source(*, sae_source: str, layer: int) -> Concept:
    base = _concepts()[1]
    return Concept(
        concept_id=base.concept_id,
        feature_index=base.feature_index,
        layer=layer,
        sae_source=sae_source,
        description=base.description,
        corpus_count=base.corpus_count,
        description_source=base.description_source,
    )


def test_roundtrip_probe_identity_accepts_matching_l0_tier() -> None:
    concept = _concept_with_sae_source(
        sae_source="22-gemmascope-2-res-16k-medium",
        layer=22,
    )

    _validate_roundtrip_probe_matches_concepts(
        candidate_rows=[_candidate()],
        concepts_by_id={1: concept},
        probe_config=RoundtripProbeConfig(
            sae_release="gemmascope_2",
            layer=22,
            l0=L0Tier.medium,
        ),
    )


@pytest.mark.parametrize(
    ("sae_source", "layer", "configured_l0", "expected_sides"),
    [
        (
            "22-gemmascope-2-res-16k-medium",
            22,
            L0Tier.big,
            ("medium", "big"),
        ),
        (
            "22-gemmascope-2-res-16k-medium",
            22,
            42,
            ("medium", "42"),
        ),
        (
            "8-gemmascope-res-16k",
            8,
            L0Tier.big,
            ("None", "big"),
        ),
    ],
)
def test_roundtrip_probe_identity_rejects_l0_identity_mismatches(
    sae_source: str,
    layer: int,
    configured_l0: int | L0Tier,
    expected_sides: tuple[str, str],
) -> None:
    concept = _concept_with_sae_source(sae_source=sae_source, layer=layer)
    release = "gemmascope_2" if "gemmascope-2" in sae_source else "gemmascope"

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _validate_roundtrip_probe_matches_concepts(
            candidate_rows=[_candidate()],
            concepts_by_id={1: concept},
            probe_config=RoundtripProbeConfig(
                sae_release=release,
                layer=layer,
                l0=configured_l0,
            ),
        )

    message = str(exc_info.value)
    assert "l0 tier identity" in message
    for side in expected_sides:
        assert side in message


@pytest.mark.parametrize(
    ("concept_updates", "probe_config", "match"),
    [
        ({}, RoundtripProbeConfig(sae_release="gemmascope_2"), "release"),
        ({}, RoundtripProbeConfig(activation_type="mlp"), "activation"),
        ({}, RoundtripProbeConfig(sae_width="32k"), "width"),
        (
            {"sae_source": "22-gemmascope-2-res-16k-medium"},
            RoundtripProbeConfig(),
            "l0 tier",
        ),
    ],
)
def test_roundtrip_probe_identity_rejects_sae_metadata_mismatches(
    concept_updates: dict[str, Any],
    probe_config: RoundtripProbeConfig,
    match: str,
) -> None:
    base = _concepts()[1]
    concepts = {
        1: Concept(
            concept_id=base.concept_id,
            feature_index=base.feature_index,
            layer=base.layer,
            sae_source=concept_updates.get("sae_source", base.sae_source),
            description=base.description,
            corpus_count=base.corpus_count,
            description_source=base.description_source,
        )
    }

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _validate_roundtrip_probe_matches_concepts(
            candidate_rows=[_candidate()],
            concepts_by_id=concepts,
            probe_config=probe_config,
        )


def test_roundtrip_probe_identity_validates_all_candidate_concepts() -> None:
    good = _concepts()[1]
    bad = Concept(
        concept_id=2,
        feature_index=43,
        layer=8,
        sae_source="8-gemmascope-mlp-16k",
        description="bad source",
        corpus_count=5,
        description_source="fixture",
    )
    row0 = _candidate(candidate_id="c0")
    row1 = _candidate(candidate_id="c1")
    row1["concept_id"] = 2

    with pytest.raises(AxBenchBpbAdvError, match="concept_id=2"):
        _validate_roundtrip_probe_matches_concepts(
            candidate_rows=[row0, row1],
            concepts_by_id={1: good, 2: bad},
            probe_config=RoundtripProbeConfig(),
        )


@pytest.mark.parametrize(
    ("layer", "n_sae_features", "l0_tier", "match"),
    [
        (9, 16384, None, "stats layer"),
        (8, 32768, None, "n_sae_features"),
        (8, 16384, L0Tier.medium, "l0 tier"),
    ],
)
def test_roundtrip_stats_metadata_must_match_probe(
    layer: int,
    n_sae_features: int,
    l0_tier: Any,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load_statistics(_path: str) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(
                layer=layer,
                n_sae_features=n_sae_features,
            ),
            l0_tier=l0_tier,
        )

    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_statistics",
        _fake_load_statistics,
    )

    with pytest.raises(AxBenchBpbAdvError, match=match):
        _validate_roundtrip_stats_match_probe(
            stats_path="/tmp/stats.safetensors",
            probe_config=RoundtripProbeConfig(),
        )


def test_roundtrip_stats_metadata_accepts_matching_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load_statistics(_path: str) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(layer=8, n_sae_features=16384),
            l0_tier=None,
        )

    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_statistics",
        _fake_load_statistics,
    )

    _validate_roundtrip_stats_match_probe(
        stats_path="/tmp/stats.safetensors",
        probe_config=RoundtripProbeConfig(),
    )


@pytest.mark.parametrize(
    ("stats_l0_tier", "configured_l0", "expected_sides"),
    [
        (L0Tier.medium, L0Tier.big, ("medium", "big")),
        (L0Tier.medium, 42, ("medium", "None")),
        (None, L0Tier.big, ("None", "big")),
    ],
)
def test_roundtrip_stats_metadata_rejects_l0_tier_mismatches(
    stats_l0_tier: L0Tier | None,
    configured_l0: int | L0Tier,
    expected_sides: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_load_statistics(_path: str) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(layer=8, n_sae_features=16384),
            l0_tier=stats_l0_tier,
        )

    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_statistics",
        _fake_load_statistics,
    )

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        _validate_roundtrip_stats_match_probe(
            stats_path="/tmp/stats.safetensors",
            probe_config=RoundtripProbeConfig(l0=configured_l0),
        )

    message = str(exc_info.value)
    assert "stats l0 tier" in message
    for side in expected_sides:
        assert side in message


@pytest.mark.parametrize(
    "configured_l0",
    [L0Tier.medium, 42],
)
def test_roundtrip_stats_metadata_accepts_matching_tier_or_numeric_v1(
    configured_l0: int | L0Tier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats_l0_tier = (
        configured_l0 if isinstance(configured_l0, L0Tier) else None
    )

    def _fake_load_statistics(_path: str) -> Any:
        return SimpleNamespace(
            metadata=SimpleNamespace(layer=8, n_sae_features=16384),
            l0_tier=stats_l0_tier,
        )

    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_statistics",
        _fake_load_statistics,
    )

    _validate_roundtrip_stats_match_probe(
        stats_path="/tmp/stats.safetensors",
        probe_config=RoundtripProbeConfig(l0=configured_l0),
    )


def _patch_roundtrip_checkpoint_stage(
    monkeypatch: pytest.MonkeyPatch,
    *,
    events: list[str] | None = None,
    segment_size: int = 384,
) -> None:
    store = SimpleNamespace(
        expected_count=1,
        restored_generation_count=0,
        restored_probe_count=0,
        pending_count=1,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "resolve_and_pin_target_checkpoints",
        lambda _cfg, _targets: {},
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "parse_feature_dir",
        lambda _path: SimpleNamespace(segment_size=segment_size),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "candidates_payload_sha256",
        lambda **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "build_roundtrip_checkpoint_fingerprint",
        lambda **_kwargs: {"identity": "test"},
    )
    monkeypatch.setattr(
        roundtrip_scoring.RoundtripCheckpointStore,
        "prepare",
        lambda **_kwargs: (
            events.append("prepare_checkpoint") if events is not None else None
        )
        or store,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "current_runtime_fingerprint",
        lambda _device: ({"runtime": "cpu"}, "1" * 64),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "warn_on_roundtrip_runtime_mismatch",
        lambda _store, _family, _runtime: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "assert_target_checkpoints_unchanged",
        lambda _targets: None,
    )


def test_model_backed_roundtrip_score_stage_offloads_before_probe_load(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _patch_roundtrip_checkpoint_stage(monkeypatch, events=events)
    candidates = _table([_candidate()])
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        target_models=["decoder_only"],
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        offload_between_stages=False,
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    write_evaluation_population(
        paths.evaluation_population(),
        candidates.to_pylist(),
        ours=cfg.ours_gcg,
    )

    class _FakeRuntimeState:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def decoder_only(self) -> Any:
            events.append("load_target")
            return SimpleNamespace(
                model=object(),
                tokenizer=_TinyTokenizer(),
                max_seq_len=128,
            )

        def target_device(self) -> str:
            return "cpu"

        def offload_models(self) -> None:
            events.append("offload")

    def _fake_bpb_scores(**kwargs: Any) -> pa.Table:
        del kwargs
        events.append("bpb_score")
        return pa.table(
            {
                "candidate_id": ["c0"],
                "loss_concept_rich": [1.25],
                "loss_concept_avoiding": [3.5],
            }
        )

    def _fake_generation_rows(**kwargs: Any) -> list[dict[str, Any]]:
        events.append("generate")
        assert kwargs["baseline_losses_by_candidate_id"] == {"c0": (1.25, 3.5)}
        assert kwargs["reference_segment_size_by_target"] == {"decoder_only": 384}
        return [{"candidate_id": "c0"}]

    def _fake_score_rows(**kwargs: Any) -> pa.Table:
        events.append("probe_score")
        assert kwargs["generation_rows"] == [{"candidate_id": "c0"}]
        assert kwargs["probe_model"] == "probe"
        return pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA)

    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.RuntimeState",
        _FakeRuntimeState,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring._read_concepts",
        lambda _source_paths: _concepts(),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring._build_token_bytes",
        lambda _tokenizer: {},
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.detect_target_model_metadata",
        lambda _model, _cfg: _target_meta(segment_size=384),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.ModelBackedDualCompletionScorer",
        lambda **_kwargs: SimpleNamespace(concepts_by_id=_concepts()),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_preflight_direction_scales_for_rows",
        lambda **_kwargs: SimpleNamespace(identity="dirmaxact:test"),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.build_bpb_adv_score_table",
        _fake_bpb_scores,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.build_roundtrip_generation_rows",
        _fake_generation_rows,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring._validate_roundtrip_stats_match_probe",
        lambda **_kwargs: events.append("validate_stats"),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_percentile_table_for_roundtrip",
        lambda _stats_path: (None, "checksum"),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.load_probe_model_for_roundtrip",
        lambda _probe: events.append("load_probe") or "probe",
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb_adv.roundtrip_scoring.score_roundtrip_generation_rows",
        _fake_score_rows,
    )

    out = run_model_backed_roundtrip_score_stage(
        cfg=cfg,
        paths=paths,
        candidates=candidates,
    )

    assert out.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert events == [
        "prepare_checkpoint",
        "load_target",
        "bpb_score",
        "generate",
        "offload",
        "validate_stats",
        "load_probe",
        "probe_score",
    ]


@pytest.mark.parametrize("target_model", ["encdec", "decoder_only"])
def test_clustered_roundtrip_stage_threads_contract_on_every_target_branch(
    target_model: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_roundtrip_checkpoint_stage(monkeypatch)
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)
    candidates = _table([_candidate(target_model=target_model)])
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        target_models=cast(Any, [target_model]),
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    observed: dict[str, Any] = {}
    validated_concepts = _concepts()
    source_identity = object()
    stage_source = SimpleNamespace(
        validation=SimpleNamespace(
            concept_entries=lambda: [
                asdict(concept)
                for concept in validated_concepts.values()
            ]
        ),
        source_identity=source_identity,
    )
    candidate_snapshot = SimpleNamespace(payload=b"candidate-bytes", sha256="a" * 64)

    def _require_stage_source(**kwargs: Any) -> object:
        observed["stage_source_guard"] = kwargs
        return stage_source

    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_stage_source",
        _require_stage_source,
    )
    def _require_candidate_snapshot(snapshot: Any, **kwargs: Any) -> Any:
        observed["candidate_snapshot_guard"] = {"snapshot": snapshot, **kwargs}
        return snapshot

    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_candidate_snapshot",
        _require_candidate_snapshot,
    )

    def _read_population(_path: Any, **kwargs: Any) -> object:
        observed["population"] = kwargs
        return object()

    def _coverage(table: pa.Table, **kwargs: Any) -> pa.Table:
        observed["coverage"] = kwargs
        return table

    monkeypatch.setattr(roundtrip_scoring, "read_evaluation_population", _read_population)
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_precomputed_bag_candidates_cover_scored_prefixes",
        _coverage,
    )

    class _FakeRuntimeState:
        def __init__(self, **kwargs: Any) -> None:
            observed["runtime"] = kwargs

        def decoder_only(self) -> Any:
            return SimpleNamespace(
                model=object(),
                tokenizer=_TinyTokenizer(),
                max_seq_len=128,
                cluster_contract=contract,
            )

        def encdec(self) -> Any:
            return SimpleNamespace(
                model=object(),
                tokenizer=_TinyTokenizer(),
                max_seq_len=128,
                cluster_contract=contract,
            )

        def target_device(self) -> str:
            return "cpu"

        def inactive_sampler(self) -> None:
            return None

        def offload_models(self) -> None:
            observed["offloaded"] = True

    monkeypatch.setattr(roundtrip_scoring, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(
        roundtrip_scoring,
        "_read_concepts",
        lambda _paths: pytest.fail(
            "clustered roundtrip stage must not reopen concepts.json"
        ),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_validate_roundtrip_probe_matches_concepts",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(roundtrip_scoring, "_build_token_bytes", lambda _tokenizer: {})
    monkeypatch.setattr(
        roundtrip_scoring,
        "get_extraction_tokenizer",
        lambda _model_id: _TinyTokenizer(),
    )

    def _load_context(_paths: Any, **kwargs: Any) -> dict[str, Any]:
        observed["context"] = kwargs
        return {}

    monkeypatch.setattr(
        roundtrip_scoring,
        "load_context_segments_by_prefix_hash",
        _load_context,
    )

    def _detect(_model: Any, _cfg: Any, **kwargs: Any) -> TargetModelMetadata:
        observed["detect"] = kwargs
        return _target_meta(segment_size=384)

    monkeypatch.setattr(roundtrip_scoring, "detect_target_model_metadata", _detect)
    monkeypatch.setattr(
        roundtrip_scoring,
        "_preflight_direction_scales_for_rows",
        lambda **_kwargs: SimpleNamespace(identity="dirmaxact:test"),
    )

    def _scorer(**kwargs: Any) -> Any:
        observed["scorer"] = kwargs
        return SimpleNamespace(concepts_by_id=_concepts())

    monkeypatch.setattr(
        roundtrip_scoring,
        "ModelBackedDualCompletionScorer",
        _scorer,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "build_bpb_adv_score_table",
        lambda **_kwargs: pa.table(
            {
                "candidate_id": ["c0"],
                "loss_concept_rich": [1.0],
                "loss_concept_avoiding": [2.0],
            }
        ),
    )

    def _generation_rows(**kwargs: Any) -> list[dict[str, Any]]:
        observed["generation"] = kwargs
        return [{"candidate_id": "c0"}]

    monkeypatch.setattr(
        roundtrip_scoring,
        "build_roundtrip_generation_rows",
        _generation_rows,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_validate_roundtrip_stats_match_probe",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "load_percentile_table_for_roundtrip",
        lambda _path: (None, "checksum"),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "load_probe_model_for_roundtrip",
        lambda _probe: object(),
    )
    expected = pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA)
    monkeypatch.setattr(
        roundtrip_scoring,
        "score_roundtrip_generation_rows",
        lambda **_kwargs: expected,
    )

    assert (
        run_model_backed_roundtrip_score_stage(
            cfg=cfg,
            paths=paths,
            candidates=candidates,
            cluster_contract=contract,
            stage_source=stage_source,  # ty: ignore[invalid-argument-type]
            candidate_snapshot=cast(Any, candidate_snapshot),
        )
        is expected
    )
    assert observed["population"] == {
        "expected_feature_contract_fingerprint": contract.feature_contract_fingerprint
    }
    assert observed["coverage"]["feature_contract_fingerprint"] == (
        contract.feature_contract_fingerprint
    )
    assert observed["runtime"]["cluster_contract"] is contract
    assert observed["stage_source_guard"] == {
        "stage_source": stage_source,
        "contract": contract,
        "surface": "roundtrip score stage",
    }
    assert observed["candidate_snapshot_guard"] == {
        "snapshot": candidate_snapshot,
        "paths": paths,
        "contract": contract,
        "source_identity": source_identity,
        "surface": "roundtrip score stage",
        "table": candidates,
    }
    assert observed["detect"]["cluster_contract"] is contract
    assert observed["scorer"]["cluster_contract"] is contract
    assert observed["scorer"]["source_validation"] is stage_source.validation
    assert observed["scorer"]["concepts_by_id"] == validated_concepts
    assert observed["generation"]["cluster_contract"] is contract
    if target_model == "encdec":
        assert observed["context"]["cluster_contract"] is contract
    else:
        assert "context" not in observed


def test_clustered_roundtrip_stage_requires_source_before_input_checks(
    tmp_path,
) -> None:
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="roundtrip score stage requires one resolved stage source",
    ):
        run_model_backed_roundtrip_score_stage(
            cfg=AxBenchBpbAdvConfig(run_tag=None),
            paths=AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run"),
            candidates=pa.table({}),
            cluster_contract=contract,
        )


def _provider_content(
    inputs: RoundtripGenerationInputs,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...], int, int], ...]:
    assert inputs.context_provider is not None
    segments = inputs.context_provider.get(inputs.prompt_tokens)
    assert segments is not None
    return tuple(
        (
            tuple(int(feature_id) for feature_id in segment.feature_ids),
            tuple(int(bucket) for bucket in segment.buckets),
            int(segment.char_start),
            int(segment.char_end),
        )
        for segment in segments
    )


def _build_context_mode_pair(
    *,
    source_segments: tuple[ContextFeatureSegment, ...],
) -> tuple[list[dict[str, Any]], list[tuple[str, tuple[Any, ...]]]]:
    rows = [
        _candidate(
            candidate_id="fixed",
            attack_row_id="fixed-row",
            target_model="encdec",
            context_signal_mode="fixed_context",
        ),
        _candidate(
            candidate_id="raw",
            attack_row_id="raw-row",
            target_model="encdec",
            context_signal_mode="not_applicable",
        ),
    ]
    control_calls: list[tuple[str, tuple[Any, ...]]] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del generation_cfg
        content = _provider_content(inputs)
        if intervention.feature_conditions == {}:
            control_calls.append((str(candidate_row["candidate_id"]), content))
        return [
            RoundtripGeneratedCompletion(
                text=repr(content),
                token_count=1,
                stop_reason="eos",
            )
            for _sample_index, _seed in sample_seeds
        ]

    output_rows = build_roundtrip_generation_rows(
        candidates=_table(rows),
        context_segments_by_prefix_hash={"hash0": source_segments},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"encdec": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"fixed": (1.0, 3.0), "raw": (1.0, 3.0)},
        build_intervention_for_row=lambda _row: Intervention.ours(
            feature_conditions={42: 1}
        ),
        build_control_intervention_for_row=lambda _row: Intervention.ours(
            feature_conditions={}
        ),
        generate_completions=_generate,
    )
    return output_rows, control_calls


def test_control_generation_does_not_share_different_resolved_provider_content() -> None:
    source_segments = (
        ContextFeatureSegment(
            feature_ids=(11,),
            buckets=(2,),
            char_start=1,
            char_end=2,
        ),
    )

    rows, control_calls = _build_context_mode_pair(source_segments=source_segments)

    assert control_calls == [
        ("fixed", (((11,), (2,), 0, 2),)),
        ("raw", (((11,), (2,), 1, 2),)),
    ]
    assert [row["control_generated_text"] for row in rows] == [
        repr(control_calls[0][1]),
        repr(control_calls[1][1]),
    ]


def test_control_generation_shares_identical_resolved_provider_content() -> None:
    source_segments = (
        ContextFeatureSegment(
            feature_ids=(11,),
            buckets=(2,),
            char_start=0,
            char_end=2,
        ),
    )

    rows, control_calls = _build_context_mode_pair(source_segments=source_segments)

    assert control_calls == [("fixed", (((11,), (2,), 0, 2),))]
    assert rows[0]["control_generated_text"] == rows[1]["control_generated_text"]


def test_control_cache_hit_returns_current_inputs_object() -> None:
    row = _candidate(target_model="encdec")
    segments = (
        ContextFeatureSegment(
            feature_ids=(11,),
            buckets=(2,),
            char_start=0,
            char_end=2,
        ),
    )
    first_inputs = RoundtripGenerationInputs(
        prompt_tokens=[97, 98],
        prompt_text="ab",
        context_provider=SingleEntryContextProvider([97, 98], segments),
    )
    current_inputs = RoundtripGenerationInputs(
        prompt_tokens=[97, 98],
        prompt_text="ab",
        context_provider=SingleEntryContextProvider([97, 98], segments),
    )
    generation_cache: roundtrip_scoring.RoundtripGenerationCache = {}
    kwargs = {
        "side": "control",
        "candidate_row": row,
        "candidate_id": "c0",
        "intervention": Intervention.ours(feature_conditions={}),
        "generation_cfg": GenerationConfig(max_new_tokens=2),
        "samples_per_candidate": 1,
        "master_seed": 123,
        "generation_cache": generation_cache,
    }
    _resolve_candidate_side_generations(
        **kwargs,
        inputs=first_inputs,
        generate_completions=lambda **_kwargs: [
            RoundtripGeneratedCompletion("cached", 1, "eos")
        ],
    )

    cached = _resolve_candidate_side_generations(
        **kwargs,
        inputs=current_inputs,
        generate_completions=lambda **_kwargs: pytest.fail(
            "identical provider content should hit the control cache"
        ),
    )

    assert cached[0][0] is current_inputs


def test_control_cache_hit_rejects_provider_content_mismatching_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _candidate()
    inputs = RoundtripGenerationInputs([97, 98], "ab", None)
    generation_cfg = GenerationConfig(max_new_tokens=2)
    identity = list(
        _control_generation_identity_key(
            row,
            control_intervention=Intervention.none(),
            control_inputs=inputs,
            sample_index=0,
            generation_cfg=generation_cfg,
        )
    )
    identity[roundtrip_scoring._CONTROL_PROVIDER_CONTENT_KEY_INDEX] = (
        "wrong-provider",
    )
    mismatched_identity = tuple(identity)
    seed = _roundtrip_seed(identity_key=mismatched_identity, master_seed=123)
    generation_cache = {
        mismatched_identity: (
            inputs,
            RoundtripGeneratedCompletion("cached", 1, "eos"),
            seed,
        )
    }
    monkeypatch.setattr(
        roundtrip_scoring,
        "_control_generation_identity_key",
        lambda *_args, **_kwargs: mismatched_identity,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="resolved context-provider content does not match",
    ):
        _resolve_candidate_side_generations(
            side="control",
            candidate_row=row,
            candidate_id="c0",
            intervention=Intervention.none(),
            inputs=inputs,
            generation_cfg=generation_cfg,
            samples_per_candidate=1,
            master_seed=123,
            generation_cache=generation_cache,
            generate_completions=lambda **_kwargs: pytest.fail(
                "mismatched cache entry reached generation"
            ),
        )


@pytest.mark.parametrize(
    ("n_candidates", "samples_per_candidate"),
    [(2, 3), (3, 1)],
    ids=["two-twins-three-samples", "three-twins-one-sample"],
)
def test_generation_rows_deduplicate_row_id_twins_but_write_each_candidate(
    monkeypatch: pytest.MonkeyPatch,
    n_candidates: int,
    samples_per_candidate: int,
) -> None:
    rows = [
        _candidate(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(n_candidates)
    ]
    batch_calls: list[tuple[str, list[tuple[int, int]]]] = []

    def generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del intervention, inputs, generation_cfg
        batch_calls.append((str(candidate_row["candidate_id"]), list(sample_seeds)))
        return [
            RoundtripGeneratedCompletion(" generated", 3, "max_tokens")
            for _ in sample_seeds
        ]

    counter_calls: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        roundtrip_scoring,
        "record_counter",
        lambda name, value=1, **attrs: counter_calls.append((name, value, attrs)),
    )
    stats: dict[str, int] = {}

    out = build_roundtrip_generation_rows(
        candidates=_table(rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=samples_per_candidate,
        master_seed=123,
        baseline_losses_by_candidate_id={
            str(row["candidate_id"]): (1.0, 3.0) for row in rows
        },
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=generate,
        stats=stats,
    )

    assert [(row["candidate_id"], row["sample_index"]) for row in out] == [
        (candidate_id, sample_index)
        for candidate_id in (f"c{i}" for i in range(n_candidates))
        for sample_index in range(samples_per_candidate)
    ]
    assert {row["generated_text"] for row in out} == {" generated"}
    assert {row["control_generated_text"] for row in out} == {" generated"}
    assert {row["target_feature_id"] for row in out} == {42}
    assert [candidate_id for candidate_id, _sample_seeds in batch_calls] == ["c0", "c0"]
    assert all(
        len(sample_seeds) == samples_per_candidate
        for _candidate_id, sample_seeds in batch_calls
    )
    request_sides = [
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.sample_requests" and value == 1
    ]
    assert Counter(request_sides) == {
        "steered": n_candidates * samples_per_candidate,
        "control": n_candidates * samples_per_candidate,
    }
    cache_hits = [
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.sample_cache_hits" and value == 1
    ]
    expected_hits_per_side = (n_candidates - 1) * samples_per_candidate
    assert Counter(cache_hits) == {
        "steered": expected_hits_per_side,
        "control": expected_hits_per_side,
    }
    assert [
        (name, value)
        for name, value, _attrs in counter_calls
        if name
        in {
            "axbench_bpb_adv.roundtrip.control_unique_batches",
            "axbench_bpb_adv.roundtrip.control_unique_completions",
        }
    ] == [
        ("axbench_bpb_adv.roundtrip.control_unique_batches", 1),
        (
            "axbench_bpb_adv.roundtrip.control_unique_completions",
            samples_per_candidate,
        ),
    ]
    assert stats == {
        "generated_unique": samples_per_candidate,
        "generation_cache_hits": expected_hits_per_side,
        "generated_tokens": 3 * n_candidates * samples_per_candidate,
        "control_generated_unique": samples_per_candidate,
        "control_generation_cache_hits": expected_hits_per_side,
        "control_generated_tokens": 3 * n_candidates * samples_per_candidate,
    }


def test_canonical_provider_content_requires_prompt_tokens_in_provider() -> None:
    """An encdec provider that cannot resolve its own generation prompt is a
    corrupt-inputs state and must fail loudly, never key on partial content."""

    class _EmptyProvider:
        def get(self, _token_ids):
            return None

    inputs = RoundtripGenerationInputs(
        [97, 98],
        "ab",
        _EmptyProvider(),  # ty: ignore[invalid-argument-type]
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="context provider does not contain its generation prompt tokens",
    ):
        roundtrip_scoring._canonical_resolved_provider_content(inputs)


def test_control_cache_hit_rejects_key_without_provider_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached control key predating the provider-content component must be
    rejected loudly on hit validation, never silently reused."""

    row = _candidate()
    inputs = RoundtripGenerationInputs([97, 98], "ab", None)
    generation_cfg = GenerationConfig(max_new_tokens=2)
    truncated_identity = tuple(
        _control_generation_identity_key(
            row,
            control_intervention=Intervention.none(),
            control_inputs=inputs,
            sample_index=0,
            generation_cfg=generation_cfg,
        )
    )[: roundtrip_scoring._CONTROL_PROVIDER_CONTENT_KEY_INDEX]
    seed = _roundtrip_seed(identity_key=truncated_identity, master_seed=123)
    generation_cache = {
        truncated_identity: (
            inputs,
            RoundtripGeneratedCompletion("cached", 1, "eos"),
            seed,
        )
    }
    monkeypatch.setattr(
        roundtrip_scoring,
        "_control_generation_identity_key",
        lambda *_args, **_kwargs: truncated_identity,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="cached control generation key has no resolved context-provider",
    ):
        _resolve_candidate_side_generations(
            side="control",
            candidate_row=row,
            candidate_id="c0",
            intervention=Intervention.none(),
            inputs=inputs,
            generation_cfg=generation_cfg,
            samples_per_candidate=1,
            master_seed=123,
            generation_cache=generation_cache,
            generate_completions=lambda **_kwargs: pytest.fail(
                "truncated cache entry reached generation"
            ),
        )


def test_generation_rows_share_semantic_controls_across_coefficients_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _candidate(candidate_id="c0", attack_row_id="r0")
    second = _candidate(candidate_id="c1", attack_row_id="r1")
    second["coefficient_str"] = "2.0"
    rows = [first, second]
    calls: list[tuple[str, str, list[tuple[int, int]]]] = []

    def _build(row: Mapping[str, Any]) -> Intervention:
        return Intervention.from_direction(
            kind="diffmean",
            target_model="decoder_only",
            direction=t.tensor([1.0]),
            direction_layer=8,
            coefficient=float(row["coefficient_str"]),
        )

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del inputs, generation_cfg
        calls.append(
            (
                str(candidate_row["candidate_id"]),
                InterventionKind.coerce(intervention.kind).value,
                list(sample_seeds),
            )
        )
        return [
            RoundtripGeneratedCompletion(
                text=(
                    f"{InterventionKind.coerce(intervention.kind).value}:"
                    f"{candidate_row['candidate_id']}:{seed}"
                ),
                token_count=1,
                stop_reason="eos",
            )
            for _sample_index, seed in sample_seeds
        ]

    counter_calls: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        roundtrip_scoring,
        "record_counter",
        lambda name, value=1, **attrs: counter_calls.append((name, value, attrs)),
    )
    out = build_roundtrip_generation_rows(
        candidates=_table(rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0), "c1": (1.0, 3.0)},
        build_intervention_for_row=_build,
        build_control_intervention_for_row=lambda _row: Intervention.none(),
        generate_completions=_generate,
    )

    assert [(candidate_id, kind) for candidate_id, kind, _seeds in calls] == [
        ("c0", "diffmean"),
        ("c0", "none"),
        ("c1", "diffmean"),
    ]
    assert out[0]["control_generation_seed"] == out[1]["control_generation_seed"]
    assert out[0]["control_generated_text"] == out[1]["control_generated_text"]
    assert out[0]["generated_text"] != out[1]["generated_text"]
    for candidate_row, output_row in zip(rows, out, strict=True):
        candidate_cfg = GenerationConfig(
            max_new_tokens=int(output_row["reference_segment_token_count"])
        )
        expected_identity = _roundtrip_generation_batch_identity_key(
            candidate_row,
            prompt_tokens=output_row["roundtrip_prompt_tokens"],
            generation_cfg=candidate_cfg,
        )
        assert output_row["generation_seed"] == _roundtrip_batch_seed(
            identity_key_without_sample_index=expected_identity,
            master_seed=123,
        )
    assert [
        (name, value)
        for name, value, _attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.control_unique_batches"
    ] == [("axbench_bpb_adv.roundtrip.control_unique_batches", 1)]
    assert [
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.sample_cache_hits" and value == 1
    ] == ["control"]


def test_generation_rows_pin_steered_and_semantic_control_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _candidate(
        candidate_id="frozen-candidate",
        attack_row_id="frozen-attack",
        mode="positive",
        target_model="decoder_only",
        source=[97, 98],
        attacked=[97, 98],
        adversarial_strength=0.0,
        strength_index=0,
    )
    batch_calls: list[tuple[list[int], list[tuple[int, int]]]] = []

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[80],
        )

    def _build_control(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[90],
        )

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, generation_cfg
        batch_calls.append((list(inputs.prompt_tokens), list(sample_seeds)))
        return [
            RoundtripGeneratedCompletion(f"sample={index}", 1, "bos_token")
            for index, _seed in sample_seeds
        ]

    counter_calls: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        roundtrip_scoring,
        "record_counter",
        lambda name, value=1, **attrs: counter_calls.append((name, value, attrs)),
    )

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=4,
        master_seed=123,
        baseline_losses_by_candidate_id={"frozen-candidate": (1.0, 3.0)},
        build_intervention_for_row=_build,
        build_control_intervention_for_row=_build_control,
        generate_completions=_generate,
    )

    # PR7 derives one seed from each identity payload with sample_index removed.
    # These literals freeze that derivation without circular helper expectations.
    expected_steered_sample_seeds = [
        (0, 567016286),
        (1, 567016286),
        (2, 567016286),
        (3, 567016286),
    ]
    expected_control_sample_seeds = [
        (0, 1431796564),
        (1, 1431796564),
        (2, 1431796564),
        (3, 1431796564),
    ]
    assert batch_calls == [
        ([80, *row["attacked_prompt_tokens"]], expected_steered_sample_seeds),
        ([90, *row["attacked_prompt_tokens"]], expected_control_sample_seeds),
    ]
    assert [row["generation_seed"] for row in out] == [
        seed for _sample_index, seed in expected_steered_sample_seeds
    ]
    assert [row["control_generation_seed"] for row in out] == [
        seed for _sample_index, seed in expected_control_sample_seeds
    ]
    request_sides = [
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.sample_requests" and value == 1
    ]
    assert Counter(request_sides) == {"steered": 4, "control": 4}
    assert not any(
        name == "axbench_bpb_adv.roundtrip.sample_cache_hits"
        for name, _value, _attrs in counter_calls
    )


def test_generation_rows_rejects_mixed_candidate_cache_state() -> None:
    row = _candidate()
    candidate_cfg = GenerationConfig(max_new_tokens=2)
    inputs = RoundtripGenerationInputs(
        prompt_tokens=list(row["attacked_prompt_tokens"]),
        prompt_text=str(row["attacked_prompt_text"]),
        context_provider=None,
    )
    identity = _roundtrip_generation_identity_key(
        row,
        prompt_tokens=inputs.prompt_tokens,
        sample_index=0,
        generation_cfg=candidate_cfg,
    )
    seed = _roundtrip_seed(identity_key=identity, master_seed=123)
    generation_cache = {
        identity: (
            inputs,
            RoundtripGeneratedCompletion("cached", 1, "bos_token"),
            seed,
        )
    }

    with pytest.raises(
        AxBenchBpbAdvError,
        match=(
            r"candidate_id='c0' has mixed steered generation cache state: "
            r"cached sample indices \[0\]; missing sample indices \[1\]"
        ),
    ):
        _resolve_candidate_side_generations(
            side="steered",
            candidate_row=row,
            candidate_id="c0",
            intervention=Intervention.none(),
            inputs=inputs,
            generation_cfg=candidate_cfg,
            samples_per_candidate=2,
            master_seed=123,
            generation_cache=generation_cache,
            generate_completions=lambda **_kwargs: pytest.fail(
                "mixed cache state reached generation"
            ),
        )


def test_generation_rows_rejects_batch_result_count_mismatch() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"candidate_id='c0' steered batch generation returned 0 completions for 2",
    ):
        build_roundtrip_generation_rows(
            candidates=_table([_candidate()]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 256},
            samples_per_candidate=2,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=lambda **_kwargs: [],
        )


def test_generation_rows_preserve_candidate_then_sample_order() -> None:
    candidate_rows = [
        _candidate(candidate_id="second", attack_row_id="r1"),
        _candidate(candidate_id="first", attack_row_id="r0", source=[120, 121]),
    ]

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del intervention, inputs, generation_cfg
        return [
            RoundtripGeneratedCompletion(
                f"{candidate_row['candidate_id']}:{sample_index}",
                1,
                "bos_token",
            )
            for sample_index, _seed in sample_seeds
        ]

    rows = build_roundtrip_generation_rows(
        candidates=_table(candidate_rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=3,
        master_seed=123,
        baseline_losses_by_candidate_id={
            str(row["candidate_id"]): (1.0, 3.0) for row in candidate_rows
        },
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=_generate,
    )

    expected = [
        (str(candidate["candidate_id"]), sample_index)
        for candidate in candidate_rows
        for sample_index in range(3)
    ]
    assert [(row["candidate_id"], row["sample_index"]) for row in rows] == expected


def test_generation_rows_require_bpb_baseline_loss_for_every_candidate() -> None:
    row = _candidate()
    generate, _calls = _generate_text()

    with pytest.raises(AxBenchBpbAdvError, match="missing BPB baseline losses"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 256},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=generate,
        )


def test_generation_rows_sample_reference_length_for_candidate_mode() -> None:
    positive = _candidate(
        candidate_id="positive",
        mode="positive",
        source=[97, 98, 99, 100, 101],
    )
    positive["concept_rich_output_tokens"] = [10, 11, 12, 13, 14]
    positive["concept_avoiding_output_tokens"] = [20]
    negative = _candidate(
        candidate_id="negative",
        mode="negative",
        source=[97, 98, 99, 100, 101, 102],
    )
    negative["concept_rich_output_tokens"] = [30]
    negative["concept_avoiding_output_tokens"] = [40, 41, 42, 43]
    calls: list[tuple[str, int]] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del intervention, inputs
        results = []
        for _sample_index, _seed in sample_seeds:
            calls.append(
                (
                    str(candidate_row["candidate_id"]),
                    int(generation_cfg.max_new_tokens),
                )
            )
            results.append(
                RoundtripGeneratedCompletion(
                    text="x" * int(generation_cfg.max_new_tokens),
                    token_count=int(generation_cfg.max_new_tokens),
                    stop_reason="max_tokens",
                )
            )
        return results

    out = build_roundtrip_generation_rows(
        candidates=_table([positive, negative]),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=999),
        reference_segment_size_by_target={"decoder_only": 4},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={
            "positive": (1.0, 3.0),
            "negative": (4.0, 2.0),
        },
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=_generate,
    )

    assert calls == [
        ("positive", 3),
        ("positive", 3),
        ("negative", 2),
        ("negative", 2),
    ]
    assert [row["reference_segment_token_count"] for row in out] == [3, 2]
    assert [row["n_generated_tokens"] for row in out] == [3, 2]
    assert [row["control_n_generated_tokens"] for row in out] == [3, 2]


def test_generation_rows_budget_uses_extraction_segment_boundary() -> None:
    row = _candidate(source=[97, 98, 99, 100, 101])
    row["concept_rich_output_tokens"] = [102, 103, 104, 105, 106]
    calls: list[int] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, inputs
        calls.extend(int(generation_cfg.max_new_tokens) for _ in sample_seeds)
        return [
            RoundtripGeneratedCompletion(
                text="x" * int(generation_cfg.max_new_tokens),
                token_count=int(generation_cfg.max_new_tokens),
                stop_reason="max_tokens",
            )
            for _ in sample_seeds
        ]

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_PairExtractionTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=999),
        reference_segment_size_by_target={"decoder_only": 3},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=_generate,
    )

    assert calls == [1, 1]
    assert out[0]["reference_segment_token_count"] == 1


def test_generation_rows_budget_uses_source_prompt_token_context() -> None:
    row = _candidate(source=[1])
    row["prefix_text"] = "a"
    row["attacked_prompt_text"] = "a"
    row["concept_rich_output_tokens"] = [2, 3, 4]
    row["concept_rich_output_digest"] = token_sequence_digest([2, 3, 4])
    calls: list[int] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, inputs
        calls.extend(int(generation_cfg.max_new_tokens) for _ in sample_seeds)
        return [
            RoundtripGeneratedCompletion(
                text="x" * int(generation_cfg.max_new_tokens),
                token_count=int(generation_cfg.max_new_tokens),
                stop_reason="max_tokens",
            )
            for _ in sample_seeds
        ]

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash={},
        tokenizer=_BoundaryMergeTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=999),
        reference_segment_size_by_target={"decoder_only": 3},
        samples_per_candidate=2,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=_generate,
    )

    assert calls == [2, 2, 2, 2]
    assert len(out) == 2
    assert [row["sample_index"] for row in out] == [0, 1]
    assert all(row["reference_segment_token_count"] == 2 for row in out)
    assert all(row["n_generated_tokens"] == 2 for row in out)
    assert all(row["control_n_generated_tokens"] == 2 for row in out)


def test_generation_rows_caps_reference_budget_by_short_reference() -> None:
    row = _candidate(source=[97, 98, 99, 100, 101])
    row["concept_rich_output_tokens"] = [10, 11]
    generate, calls = _generate_text()

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=999),
        reference_segment_size_by_target={"decoder_only": 4},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=generate,
    )

    assert [call[-1] for call in calls] == [2, 2]
    assert out[0]["reference_segment_token_count"] == 2


def test_generation_rows_require_reference_segment_size_for_target() -> None:
    row = _candidate()
    generate, _calls = _generate_text()

    with pytest.raises(AxBenchBpbAdvError, match="reference segment size"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=generate,
        )


def test_generation_rows_reject_invalid_reference_segment_size() -> None:
    row = _candidate()
    generate, _calls = _generate_text()

    with pytest.raises(AxBenchBpbAdvError, match="reference_segment_size"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 0},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=generate,
        )


def test_generation_rows_reject_prefix_text_token_mismatch() -> None:
    row = _candidate(source=[97, 98, 99])
    row["prefix_text"] = "abd"
    generate, _calls = _generate_text()

    with pytest.raises(AxBenchBpbAdvError, match="prefix_text"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 256},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=generate,
        )


def test_generation_rows_reject_empty_reference_length() -> None:
    row = _candidate()
    row["concept_rich_output_tokens"] = []
    generate, _calls = _generate_text()

    with pytest.raises(AxBenchBpbAdvError, match="empty concept_rich_output_tokens"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 256},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build_intervention,
            build_control_intervention_for_row=_build_intervention,
            generate_completions=generate,
        )


def test_generation_rows_do_not_reuse_control_across_adversarial_strengths() -> None:
    rows = [
        _candidate(candidate_id="c0", adversarial_strength=1.0, strength_index=1),
        _candidate(candidate_id="c1", adversarial_strength=2.0, strength_index=2),
    ]
    for row in rows:
        row["adversarial_method"] = "additive_direction"
    calls: list[tuple[str, tuple[int, ...]]] = []

    def _build(row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[80 + int(row["strength_index"])],
        )

    def _build_control(row: Mapping[str, Any]) -> Intervention:
        return Intervention.none().with_counter_direction(
            direction=t.tensor([1.0]),
            layer=8,
            coefficient=-float(row["adversarial_strength"]),
        )

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del generation_cfg, inputs
        prefix = tuple(intervention.prompt_prefix_tokens or ())
        results = []
        for _sample_index, seed in sample_seeds:
            calls.append((str(candidate_row["candidate_id"]), prefix))
            results.append(
                RoundtripGeneratedCompletion(
                    text=(
                        f"prefix={prefix} counter="
                        f"{intervention.counter_direction_coefficient} seed={seed}"
                    ),
                    token_count=1,
                    stop_reason="eos",
                )
            )
        return results

    out = build_roundtrip_generation_rows(
        candidates=_table(rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={
            str(row["candidate_id"]): (1.0, 3.0) for row in rows
        },
        build_intervention_for_row=_build,
        build_control_intervention_for_row=_build_control,
        generate_completions=_generate,
    )

    assert len(calls) == 4
    assert out[0]["roundtrip_prompt_tokens"] == [81, 97, 98]
    assert out[0]["control_roundtrip_prompt_tokens"] == [97, 98]
    assert out[1]["roundtrip_prompt_tokens"] == [82, 97, 98]
    assert out[1]["control_roundtrip_prompt_tokens"] == [97, 98]
    assert out[0]["generated_text"] != out[1]["generated_text"]
    assert out[0]["control_generated_text"] != out[1]["control_generated_text"]
    assert out[0]["control_generation_seed"] != out[1]["control_generation_seed"]


def test_generation_rows_probe_prompt_text_uses_stored_attacked_text_without_bos() -> None:
    row = _candidate(source=[0, 97, 98])
    assert row["attacked_prompt_text"] == "ab"
    seen_prompt_texts: list[str] = []

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[80],
        )

    def _build_control(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.none()

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, generation_cfg
        seen_prompt_texts.extend(inputs.prompt_text for _ in sample_seeds)
        return [
            RoundtripGeneratedCompletion(
                text=f"prompt={inputs.prompt_text}",
                token_count=1,
                stop_reason="eos",
            )
            for _ in sample_seeds
        ]

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash={},
        tokenizer=_VisibleBosTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build,
        build_control_intervention_for_row=_build_control,
        generate_completions=_generate,
    )

    assert out[0]["roundtrip_prompt_tokens"] == [80, 0, 97, 98]
    assert out[0]["roundtrip_prompt_text"] == "Pab"
    assert out[0]["control_roundtrip_prompt_tokens"] == [0, 97, 98]
    assert out[0]["control_roundtrip_prompt_text"] == "ab"
    assert seen_prompt_texts == ["Pab", "ab"]
    assert "<BOS>" not in out[0]["roundtrip_prompt_text"]
    assert "<BOS>" not in out[0]["control_roundtrip_prompt_text"]


def test_generation_rows_rejects_attack_that_changes_owned_bos() -> None:
    row = _candidate(source=[1, 2, 3], attacked=[0, 2, 3])
    row["prefix_text"] = "abc"
    row["attacked_prompt_text"] = "LONGc"
    row["editable_token_start"] = 0
    row["editable_token_count"] = 1
    row["replacement_tokens"] = [0]
    row["replacement_text"] = "LONGc"
    row["concept_rich_output_tokens"] = [4]
    row["concept_rich_output_digest"] = token_sequence_digest([4])
    seen_prompt_texts: list[str] = []

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="decoder_only",
            prompt_prefix_tokens=[1],
        )

    def _build_control(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.none()

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, generation_cfg
        seen_prompt_texts.extend(inputs.prompt_text for _ in sample_seeds)
        return [
            RoundtripGeneratedCompletion(
                text=f"prompt={inputs.prompt_text}",
                token_count=1,
                stop_reason="eos",
            )
            for _ in sample_seeds
        ]

    with pytest.raises(AxBenchBpbAdvError, match="immutable BOS"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash={},
            tokenizer=_BoundaryMergeTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"decoder_only": 4},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build,
            build_control_intervention_for_row=_build_control,
            generate_completions=_generate,
        )
    assert seen_prompt_texts == []


def test_generation_rows_samples_per_candidate_are_seed_deterministic() -> None:
    row = _candidate()

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, inputs, generation_cfg
        return [
            RoundtripGeneratedCompletion(
                text=f"sample={sample_index} seed={seed}",
                token_count=1,
                stop_reason="eos",
            )
            for sample_index, seed in sample_seeds
        ]

    kwargs: dict[str, Any] = {
        "candidates": _table([row]),
        "context_segments_by_prefix_hash": {},
        "tokenizer": _TinyTokenizer(),
        "extraction_tokenizer": _TinyTokenizer(),
        "concepts_by_id": _concepts(),
        "generation_cfg": GenerationConfig(max_new_tokens=5),
        "reference_segment_size_by_target": {"decoder_only": 256},
        "samples_per_candidate": 2,
        "baseline_losses_by_candidate_id": {"c0": (1.0, 3.0)},
        "build_intervention_for_row": _build_intervention,
        "build_control_intervention_for_row": _build_intervention,
        "generate_completions": _generate,
    }

    first = build_roundtrip_generation_rows(master_seed=123, **kwargs)
    second = build_roundtrip_generation_rows(master_seed=123, **kwargs)
    changed_seed = build_roundtrip_generation_rows(master_seed=124, **kwargs)

    assert [row["sample_index"] for row in first] == [0, 1]
    assert [row["generation_seed"] for row in first] == [
        row["generation_seed"] for row in second
    ]
    assert [row["control_generation_seed"] for row in first] == [
        row["control_generation_seed"] for row in second
    ]
    assert [row["generation_seed"] for row in first] != [
        row["generation_seed"] for row in changed_seed
    ]
    assert [row["generated_text"] for row in first] == [
        row["generated_text"] for row in second
    ]
    assert [row["generated_text"] for row in first] != [
        row["generated_text"] for row in changed_seed
    ]


def test_encdec_prompting_context_provider_adds_empty_prefix_segment() -> None:
    row = _candidate(target_model="encdec")
    context_segments = {
        "hash0": (
            ContextFeatureSegment(
                feature_ids=(10,),
                buckets=(1,),
                char_start=0,
                char_end=2,
            ),
        )
    }

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=[80],
        )

    def _build_control(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.ours(feature_conditions={})

    captured_inputs: list[RoundtripGenerationInputs] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, generation_cfg
        captured_inputs.extend(inputs for _ in sample_seeds)
        return [
            RoundtripGeneratedCompletion("ok", 1, "bos_token")
            for _ in sample_seeds
        ]

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash=context_segments,
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"encdec": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build,
        build_control_intervention_for_row=_build_control,
        generate_completions=_generate,
    )

    inputs = captured_inputs[0]
    assert out[0]["roundtrip_prompt_tokens"] == [80, 97, 98]
    assert out[0]["roundtrip_prompt_text"] == "Pab"
    assert out[0]["control_roundtrip_prompt_tokens"] == [97, 98]
    assert out[0]["control_roundtrip_prompt_text"] == "ab"
    assert inputs.context_provider is not None
    segments = inputs.context_provider.get([80, 97, 98])
    assert segments == (
        ContextFeatureSegment((), (), 0, 1),
        ContextFeatureSegment((10,), (1,), 1, 3),
    )
    assert inputs.context_provider.get([97, 98]) is None


def test_clustered_generation_inputs_require_representative_context_ids() -> None:
    row = _candidate(target_model="encdec")
    context_segments = {
        "hash0": (
            ContextFeatureSegment(
                feature_ids=(10,),
                buckets=(1,),
                char_start=0,
                char_end=2,
            ),
        )
    }
    good: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=lambda _ids, **_kwargs: None
        ),
        feature_mask=SimpleNamespace(assert_no_masked=lambda _ids: None),
    )
    representative_failure: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=lambda _ids, **_kwargs: (_ for _ in ()).throw(
                ValueError("representative sentinel")
            )
        ),
        feature_mask=good.feature_mask,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="exact clustered-cache content.*representative sentinel",
    ):
        _roundtrip_generation_inputs(
            candidate_row=row,
            context_segments_by_prefix_hash=context_segments,
            tokenizer=_TinyTokenizer(),
            intervention=Intervention.none(),
            cluster_contract=representative_failure,
        )

    restored = _roundtrip_generation_inputs(
        candidate_row=row,
        context_segments_by_prefix_hash=context_segments,
        tokenizer=_TinyTokenizer(),
        intervention=Intervention.none(),
        cluster_contract=good,
    )
    assert restored.context_provider is not None


def test_clustered_generation_inputs_require_mask_allowed_context_ids() -> None:
    row = _candidate(target_model="encdec")
    context_segments = {
        "hash0": (
            ContextFeatureSegment(
                feature_ids=(10,),
                buckets=(1,),
                char_start=0,
                char_end=2,
            ),
        )
    }
    good: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=lambda _ids, **_kwargs: None
        ),
        feature_mask=SimpleNamespace(assert_no_masked=lambda _ids: None),
    )
    mask_failure: Any = SimpleNamespace(
        consolidation=good.consolidation,
        feature_mask=SimpleNamespace(
            assert_no_masked=lambda _ids: (_ for _ in ()).throw(
                ValueError("mask sentinel")
            )
        ),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="exact clustered-cache content.*mask sentinel",
    ):
        _roundtrip_generation_inputs(
            candidate_row=row,
            context_segments_by_prefix_hash=context_segments,
            tokenizer=_TinyTokenizer(),
            intervention=Intervention.none(),
            cluster_contract=mask_failure,
        )

    restored = _roundtrip_generation_inputs(
        candidate_row=row,
        context_segments_by_prefix_hash=context_segments,
        tokenizer=_TinyTokenizer(),
        intervention=Intervention.none(),
        cluster_contract=good,
    )
    assert restored.context_provider is not None


def test_decoder_only_generation_inputs_do_not_apply_cluster_context_guards() -> None:
    def fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("decoder-only early return applied context guard")

    contract: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=fail_if_called,
        ),
        feature_mask=SimpleNamespace(assert_no_masked=fail_if_called),
    )

    inputs = _roundtrip_generation_inputs(
        candidate_row=_candidate(target_model="decoder_only"),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        intervention=Intervention.none(),
        cluster_contract=contract,
    )

    assert inputs.context_provider is None


def test_generation_rows_threads_cluster_contract_to_both_input_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract: Any = SimpleNamespace(feature_contract_fingerprint="f" * 64)
    observed: list[Any] = []

    def _inputs(**kwargs: Any) -> RoundtripGenerationInputs:
        observed.append(kwargs.get("cluster_contract"))
        return RoundtripGenerationInputs(
            prompt_tokens=[97, 98],
            prompt_text="ab",
            context_provider=None,
        )

    monkeypatch.setattr(roundtrip_scoring, "_roundtrip_generation_inputs", _inputs)
    generate, _calls = _generate_text()
    rows = build_roundtrip_generation_rows(
        candidates=_table([_candidate()]),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=generate,
        cluster_contract=contract,
    )

    assert len(rows) == 1
    assert observed == [contract, contract]


@pytest.mark.parametrize(
    ("failing_guard", "message"),
    [
        ("representative", "representative sentinel"),
        ("mask", "mask sentinel"),
    ],
)
def test_model_generate_rejects_invalid_clustered_bag_before_packing(
    failing_guard: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _representatives(_ids: Any, **_kwargs: Any) -> None:
        if failing_guard == "representative":
            raise ValueError(message)

    def _mask(_ids: Any) -> None:
        if failing_guard == "mask":
            raise ValueError(message)

    contract: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=_representatives,
        ),
        feature_mask=SimpleNamespace(assert_no_masked=_mask),
    )
    bundle = SimpleNamespace(
        model=object(),
        tokenizer=_TinyTokenizer(),
        cluster_contract=contract,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_generate_samples_with_nanochat_tokens",
        lambda **_kwargs: pytest.fail("generation reached before clustered bag guard"),
    )
    generate = _build_model_generate_batch_fn(
        target_bundles=cast(Any, {"encdec": bundle}),
        device="cpu",
        lora=None,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=f"invalid clustered bag.*{message}",
    ):
        generate(
            candidate_row={"target_model": "encdec"},
            intervention=Intervention.ours(feature_conditions={10: 1}),
            inputs=RoundtripGenerationInputs([97], "a", None),
            generation_cfg=GenerationConfig(max_new_tokens=1),
            sample_seeds=[(0, 1)],
        )


def test_model_generate_accepts_valid_clustered_bag_and_calls_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    contract: Any = SimpleNamespace(
        consolidation=SimpleNamespace(
            assert_representatives_only=lambda _ids, **_kwargs: calls.append(
                "representatives"
            )
        ),
        feature_mask=SimpleNamespace(
            assert_no_masked=lambda _ids: calls.append("mask")
        ),
    )
    bundle = SimpleNamespace(
        model=object(),
        tokenizer=_TinyTokenizer(),
        cluster_contract=contract,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "apply_intervention",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_generate_samples_with_nanochat_tokens",
        lambda **_kwargs: calls.append("generate")
        or [GenerationResult(text="ok", token_count=1, stop_reason="max_tokens")],
    )
    generate = _build_model_generate_batch_fn(
        target_bundles=cast(Any, {"encdec": bundle}),
        device="cpu",
        lora=None,
    )

    results = generate(
        candidate_row={"target_model": "encdec"},
        intervention=Intervention.ours(feature_conditions={10: 1}),
        inputs=RoundtripGenerationInputs([97], "a", None),
        generation_cfg=GenerationConfig(max_new_tokens=1),
        sample_seeds=[(0, 1)],
    )

    assert results[0].text == "ok"
    assert calls == ["representatives", "mask", "generate"]

    bundle.cluster_contract = None
    calls.clear()
    results = generate(
        candidate_row={"target_model": "encdec"},
        intervention=Intervention.ours(feature_conditions={10: 1}),
        inputs=RoundtripGenerationInputs([97], "a", None),
        generation_cfg=GenerationConfig(max_new_tokens=1),
        sample_seeds=[(0, 1)],
    )
    assert results[0].text == "ok"
    assert calls == ["generate"]


def test_model_generate_batch_enters_intervention_and_builds_engine_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    counters: list[tuple[str, int | float]] = []
    profile_regions: list[str] = []

    class _FakeEngine:
        def generate_samples(self, **kwargs: Any) -> list[list[int]]:
            events.append("generate_samples")
            assert kwargs == {
                "token_ids": [97],
                "seed": 11,
                "n_samples": 2,
                "max_new_tokens": 2,
                "temperature": 0.7,
                "top_k": 50,
            }
            return [[111], [112, 113]]

    def _build_engine(*args: Any, **kwargs: Any) -> _FakeEngine:
        del args, kwargs
        events.append("build_engine")
        return _FakeEngine()

    @contextmanager
    def _apply(*args: Any, **kwargs: Any):
        del args, kwargs
        events.append("apply_enter")
        try:
            yield
        finally:
            events.append("apply_exit")

    @contextmanager
    def _profile_region(name: str, **_attrs: Any):
        profile_regions.append(name)
        yield

    monkeypatch.setattr(generation_helpers, "build_engine_for_features", _build_engine)
    monkeypatch.setattr(
        generation_helpers,
        "record_counter",
        lambda name, value=1, **_attrs: counters.append((name, value)),
    )
    monkeypatch.setattr(roundtrip_scoring, "apply_intervention", _apply)
    monkeypatch.setattr(roundtrip_scoring, "profile_region", _profile_region)
    generate = _build_model_generate_batch_fn(
        target_bundles=cast(
            Any,
            {
                "decoder_only": SimpleNamespace(
                    model=object(),
                    tokenizer=_TinyTokenizer(),
                    cluster_contract=None,
                )
            },
        ),
        device="cpu",
        lora=None,
    )

    results = generate(
        candidate_row={"candidate_id": "c0", "target_model": "decoder_only"},
        intervention=Intervention.none(),
        inputs=RoundtripGenerationInputs([97], "a", None),
        generation_cfg=GenerationConfig(
            max_new_tokens=2,
            temperature=0.7,
            top_k_sampling=50,
        ),
        sample_seeds=[(0, 11), (1, 11)],
    )

    assert events == [
        "apply_enter",
        "build_engine",
        "generate_samples",
        "apply_exit",
    ]
    assert [(result.text, result.token_count, result.stop_reason) for result in results] == [
        ("o", 1, "bos_token"),
        ("pq", 2, "max_tokens"),
    ]
    assert counters == [
        ("axbench_base.generation.requests", 2),
        ("axbench_base.generation.prompt_tokens", 1),
        ("axbench_base.generation.engine_constructions", 1),
        ("axbench_base.generation.generated_tokens", 3),
    ]
    assert profile_regions == ["axbench_bpb_adv.roundtrip.generate_completions"]


def test_model_generate_batch_rejects_mixed_batch_seeds() -> None:
    generate = _build_model_generate_batch_fn(
        target_bundles=cast(
            Any,
            {
                "decoder_only": SimpleNamespace(
                    model=object(),
                    tokenizer=_TinyTokenizer(),
                    cluster_contract=None,
                )
            },
        ),
        device="cpu",
        lora=None,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"requires one shared batch seed; observed \[11, 22\]",
    ):
        generate(
            candidate_row={"candidate_id": "c0", "target_model": "decoder_only"},
            intervention=Intervention.none(),
            inputs=RoundtripGenerationInputs([97], "a", None),
            generation_cfg=GenerationConfig(max_new_tokens=2),
            sample_seeds=[(0, 11), (1, 22)],
        )


def test_encdec_context_provider_preserves_bos_inclusive_coordinates() -> None:
    row = _candidate(target_model="encdec", source=[0, 97, 98])
    context_segments = {
        "hash0": (
            ContextFeatureSegment(
                feature_ids=(10,),
                buckets=(1,),
                char_start=0,
                char_end=len("<BOS>ab"),
            ),
        )
    }

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.ours(feature_conditions={})

    captured_inputs: list[RoundtripGenerationInputs] = []

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, generation_cfg
        captured_inputs.extend(inputs for _ in sample_seeds)
        return [RoundtripGeneratedCompletion("ok", 1, "eos") for _ in sample_seeds]

    out = build_roundtrip_generation_rows(
        candidates=_table([row]),
        context_segments_by_prefix_hash=context_segments,
        tokenizer=_VisibleBosTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"encdec": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
        build_intervention_for_row=_build,
        build_control_intervention_for_row=_build,
        generate_completions=_generate,
    )

    inputs = captured_inputs[0]
    assert out[0]["roundtrip_prompt_tokens"] == [0, 97, 98]
    assert out[0]["roundtrip_prompt_text"] == "ab"
    assert inputs.context_provider is not None
    assert inputs.context_provider.get([0, 97, 98]) == (
        ContextFeatureSegment((10,), (1,), 0, len("<BOS>ab")),
    )


def test_encdec_prompting_rejects_changed_owned_bos_before_context_projection() -> None:
    row = _candidate(target_model="encdec", source=[1, 2, 3], attacked=[0, 2, 3])
    row["prefix_text"] = "abc"
    row["attacked_prompt_text"] = "LONGc"
    row["editable_token_start"] = 0
    row["editable_token_count"] = 1
    row["replacement_tokens"] = [0]
    row["replacement_text"] = "LONGc"
    row["concept_rich_output_tokens"] = [4]
    row["concept_rich_output_digest"] = token_sequence_digest([4])
    context_segments = {
        "hash0": (
            ContextFeatureSegment(
                feature_ids=(10,),
                buckets=(1,),
                char_start=0,
                char_end=5,
            ),
        )
    }

    def _build(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.prompting(
            target_model="encdec",
            prompt_prefix_tokens=[1],
        )

    def _build_control(_row: Mapping[str, Any]) -> Intervention:
        return Intervention.ours(feature_conditions={})

    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del candidate_row, intervention, inputs, generation_cfg
        return [RoundtripGeneratedCompletion("ok", 1, "eos") for _ in sample_seeds]

    with pytest.raises(AxBenchBpbAdvError, match="immutable BOS"):
        build_roundtrip_generation_rows(
            candidates=_table([row]),
            context_segments_by_prefix_hash=context_segments,
            tokenizer=_BoundaryMergeTokenizer(),
            extraction_tokenizer=_TinyTokenizer(),
            concepts_by_id=_concepts(),
            generation_cfg=GenerationConfig(max_new_tokens=5),
            reference_segment_size_by_target={"encdec": 4},
            samples_per_candidate=1,
            master_seed=123,
            baseline_losses_by_candidate_id={"c0": (1.0, 3.0)},
            build_intervention_for_row=_build,
            build_control_intervention_for_row=_build_control,
            generate_completions=_generate,
        )


def test_score_roundtrip_rows_delegates_exact_prompt_response_and_feature() -> None:
    generation_rows = [
        {
            **_candidate(),
            "roundtrip_prompt_tokens": [97, 98],
            "roundtrip_prompt_token_digest": token_sequence_digest([97, 98]),
            "roundtrip_prompt_text": "ab",
            "control_roundtrip_prompt_tokens": [97, 120],
            "control_roundtrip_prompt_token_digest": token_sequence_digest([97, 120]),
            "control_roundtrip_prompt_text": "ax",
            "loss_concept_rich": 1.0,
            "loss_concept_avoiding": 3.0,
            "sample_index": 0,
            "reference_segment_token_count": 3,
            "generation_seed": 123,
            "generated_text": " generated",
            "n_generated_tokens": 3,
            "stop_reason": "max_tokens",
            "control_generation_seed": 124,
            "control_generated_text": " control",
            "control_n_generated_tokens": 2,
            "control_stop_reason": "eos",
            "is_control": True,
            "target_feature_id": 42,
        }
    ]
    calls: list[dict[str, Any]] = []
    stats: dict[str, int] = {}

    def _probe(
        *,
        prompt_text: str,
        response_text: str,
        feature_ids: Sequence[int],
        probe_model: Any,
        percentile_table: Any,
        feature_chunk_size: int,
    ) -> RoundtripFeatureScores:
        calls.append(
            {
                "prompt_text": prompt_text,
                "response_text": response_text,
                "feature_ids": tuple(feature_ids),
                "probe_model": probe_model,
                "percentile_table": percentile_table,
                "feature_chunk_size": feature_chunk_size,
            }
        )
        if response_text == " generated":
            return RoundtripFeatureScores(
                n_features=1,
                pre_act_mean=2.5,
                post_act_mean=1.25,
                percentile_mean=77.0,
            )
        return RoundtripFeatureScores(
            n_features=1,
            pre_act_mean=1.5,
            post_act_mean=0.75,
            percentile_mean=55.0,
        )

    scores = score_roundtrip_generation_rows(
        generation_rows=generation_rows,
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=_probe,
        stats=stats,
    )

    assert scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert stats == {
        "probe_rows": 1,
        "empty_skipped": 0,
        "empty_steered": 0,
        "empty_control": 0,
    }
    assert calls == [
        {
            "prompt_text": "ab",
            "response_text": " generated",
            "feature_ids": (42,),
            "probe_model": "probe",
            "percentile_table": None,
            "feature_chunk_size": 17,
        },
        {
            "prompt_text": "ax",
            "response_text": " control",
            "feature_ids": (42,),
            "probe_model": "probe",
            "percentile_table": None,
            "feature_chunk_size": 17,
        },
    ]
    row = scores.to_pylist()[0]
    assert row["loss_concept_rich"] == pytest.approx(1.0)
    assert row["loss_concept_avoiding"] == pytest.approx(3.0)
    assert row["roundtrip_pre_act"] == pytest.approx(2.5)
    assert row["roundtrip_post_act"] == pytest.approx(1.25)
    assert row["roundtrip_percentile"] == pytest.approx(77.0)
    assert row["n_roundtrip_features"] == 1
    assert row["control_roundtrip_pre_act"] == pytest.approx(1.5)
    assert row["control_roundtrip_post_act"] == pytest.approx(0.75)
    assert row["control_roundtrip_percentile"] == pytest.approx(55.0)
    assert row["control_n_roundtrip_features"] == 1


def test_score_roundtrip_rows_default_probe_accepts_unfrozen_sae() -> None:
    hook_name = "blocks.8.hook_resid_post"

    class _FakeProbeModel(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = t.nn.Parameter(t.zeros(1))

        def run_with_cache(self, tokens, **kwargs):
            hook = kwargs["names_filter"][0]
            activations = t.ones(
                tokens.shape[0],
                tokens.shape[1],
                2,
                dtype=t.float32,
            )
            return None, {hook: activations}

    class _FakeProbeSAE(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            W_enc = t.zeros(2, 43)
            W_enc[:, 42] = t.tensor([1.0, 2.0])
            self.W_enc = t.nn.Parameter(W_enc)
            self.b_enc = t.nn.Parameter(t.zeros(43))
            self.b_dec = t.nn.Parameter(t.zeros(2))
            self.cfg = SimpleNamespace(
                apply_b_dec_to_input=False,
                activation_fn="relu",
            )

    probe = ProbeModel(
        model=_FakeProbeModel(),
        sae=_FakeProbeSAE(),
        tokenizer=_TinyTokenizer(),
        hook_name=hook_name,
        layer=8,
    )
    scores = score_roundtrip_generation_rows(
        generation_rows=[_generation_row_for_scoring()],
        probe_model=probe,
        percentile_table=None,
        feature_chunk_size=1,
    )

    assert scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert scores.num_rows == 1
    row = scores.to_pylist()[0]
    assert row["n_roundtrip_features"] == 1
    assert math.isfinite(row["roundtrip_pre_act"])
    assert math.isfinite(row["roundtrip_post_act"])
    assert all(parameter.grad is None for parameter in probe.sae.parameters())


def test_score_roundtrip_memoizes_pairs_and_keeps_feature_ids_in_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _generation_row_for_scoring()
    duplicate = dict(first)
    duplicate["candidate_id"] = "c1"
    duplicate["attack_row_id"] = "r1"
    different_feature = dict(first)
    different_feature["candidate_id"] = "c2"
    different_feature["attack_row_id"] = "r2"
    different_feature["target_feature_id"] = 43
    calls: list[tuple[str, str, tuple[int, ...]]] = []

    def _probe(
        *,
        prompt_text: str,
        response_text: str,
        feature_ids: Sequence[int],
        probe_model: Any,
        percentile_table: Any,
        feature_chunk_size: int,
    ) -> RoundtripFeatureScores:
        del probe_model, percentile_table, feature_chunk_size
        feature_tuple = tuple(feature_ids)
        calls.append((prompt_text, response_text, feature_tuple))
        return RoundtripFeatureScores(
            n_features=len(feature_tuple),
            pre_act_mean=float(feature_tuple[0]),
            post_act_mean=float(feature_tuple[0]),
            percentile_mean=float(feature_tuple[0]),
        )

    counter_calls: list[tuple[str, int | float, dict[str, Any]]] = []
    monkeypatch.setattr(
        roundtrip_scoring,
        "record_counter",
        lambda name, value=1, **attrs: counter_calls.append((name, value, attrs)),
    )
    scores = score_roundtrip_generation_rows(
        generation_rows=[first, duplicate, different_feature],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=_probe,
    )

    assert calls == [
        ("ab", " generated", (42,)),
        ("ab", " control", (42,)),
        ("ab", " generated", (43,)),
        ("ab", " control", (43,)),
    ]
    assert [row["roundtrip_percentile"] for row in scores.to_pylist()] == [
        42.0,
        42.0,
        43.0,
    ]
    assert Counter(
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.probe_cache_hits" and value == 1
    ) == {"steered": 1, "control": 1}
    assert Counter(
        attrs["side"]
        for name, value, attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.probe_unique_texts" and value == 1
    ) == {"steered": 2, "control": 2}
    assert sum(
        int(value)
        for name, value, _attrs in counter_calls
        if name == "axbench_bpb_adv.roundtrip.probe_requests"
    ) == 6


def _recording_probe():
    """Probe stub that records each ``response_text`` it is asked to score.

    Used to assert that empty completions are NEVER handed to the probe (the
    probe has no text to score) while non-empty ones still are.
    """
    seen: list[str] = []

    def _probe(
        *,
        prompt_text: str,
        response_text: str,
        feature_ids: Sequence[int],
        probe_model: Any,
        percentile_table: Any,
        feature_chunk_size: int,
    ) -> RoundtripFeatureScores:
        seen.append(response_text)
        return RoundtripFeatureScores(
            n_features=1, pre_act_mean=1.0, post_act_mean=1.0, percentile_mean=50.0
        )

    return _probe, seen


def test_score_roundtrip_skips_empty_steered_response_instead_of_raising() -> None:
    probe, seen = _recording_probe()
    stats: dict[str, int] = {}
    scores = score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(generated_text=""),
            _generation_row_for_scoring(generated_text=" real"),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    # Only the non-empty sample is scored; the empty one is dropped (not raised).
    assert scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert scores.num_rows == 1
    assert scores.to_pylist()[0]["generated_text"] == " real"
    # The empty steered response is never sent to the probe.
    assert " real" in seen
    assert "" not in seen
    assert stats == {
        "probe_rows": 1,
        "empty_skipped": 1,
        "empty_steered": 1,
        "empty_control": 0,
    }


def test_score_roundtrip_skips_empty_control_response() -> None:
    probe, _seen = _recording_probe()
    stats: dict[str, int] = {}
    scores = score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(control_generated_text=""),
            _generation_row_for_scoring(generated_text=" real"),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    assert scores.num_rows == 1
    assert stats == {
        "probe_rows": 1,
        "empty_skipped": 1,
        "empty_steered": 0,
        "empty_control": 1,
    }


def test_score_roundtrip_counts_both_sides_empty_once_in_skip() -> None:
    probe, _seen = _recording_probe()
    stats: dict[str, int] = {}
    scores = score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(generated_text="", control_generated_text=""),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    # A sample empty on BOTH sides is skipped exactly once, but tallied on each
    # side so the per-side counts stay independently meaningful.
    assert scores.num_rows == 0
    assert stats == {
        "probe_rows": 0,
        "empty_skipped": 1,
        "empty_steered": 1,
        "empty_control": 1,
    }


def test_score_roundtrip_no_empty_completions_reports_zero() -> None:
    probe, _seen = _recording_probe()
    stats: dict[str, int] = {}
    score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(generated_text=" a"),
            _generation_row_for_scoring(generated_text=" b"),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    assert stats == {
        "probe_rows": 2,
        "empty_skipped": 0,
        "empty_steered": 0,
        "empty_control": 0,
    }


def test_score_roundtrip_all_empty_returns_empty_table_with_schema() -> None:
    probe, seen = _recording_probe()
    stats: dict[str, int] = {}
    scores = score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(generated_text=""),
            _generation_row_for_scoring(generated_text=""),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    # No rows survive, but the table still carries the roundtrip schema and the
    # probe is never invoked.
    assert scores.num_rows == 0
    assert scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert seen == []
    assert stats == {
        "probe_rows": 0,
        "empty_skipped": 2,
        "empty_steered": 2,
        "empty_control": 0,
    }


def test_score_roundtrip_empty_skip_logs_percentage(caplog) -> None:
    probe, _seen = _recording_probe()
    score_roundtrip_generation_rows(
        generation_rows=[
            _generation_row_for_scoring(generated_text=""),
            _generation_row_for_scoring(generated_text=" real"),
        ],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
    )
    messages = [r.message for r in caplog.records]
    assert any(
        "skipped 1 of 2 (50.0%)" in m and "empty" in m for m in messages
    ), messages


def test_score_roundtrip_treats_whitespace_only_completion_as_nonempty() -> None:
    # Only the empty string "" counts as empty; whitespace-only output is real
    # generated text and must still be scored. Locks the ``== ""`` predicate
    # against a future ``.strip()``-style refactor that would wrongly drop it.
    probe, seen = _recording_probe()
    stats: dict[str, int] = {}
    scores = score_roundtrip_generation_rows(
        generation_rows=[_generation_row_for_scoring(generated_text="   ")],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
        stats=stats,
    )
    assert scores.num_rows == 1
    assert "   " in seen
    assert stats["empty_skipped"] == 0


def test_score_roundtrip_no_skip_warning_when_no_empty_completions(caplog) -> None:
    probe, _seen = _recording_probe()
    score_roundtrip_generation_rows(
        generation_rows=[_generation_row_for_scoring(generated_text=" real")],
        probe_model="probe",
        percentile_table=None,
        feature_chunk_size=17,
        probe_text_fn=probe,
    )
    assert not [
        r
        for r in caplog.records
        if "skipped" in r.message and "empty" in r.message
    ]


def _build_generation_rows_for_logging(*, n_candidates: int, log_progress_every: int):
    rows = [
        _candidate(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(n_candidates)
    ]
    generate, _calls = _generate_text()
    return build_roundtrip_generation_rows(
        candidates=_table(rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        extraction_tokenizer=_TinyTokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=5),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=1,
        master_seed=123,
        baseline_losses_by_candidate_id={
            str(row["candidate_id"]): (1.0, 3.0) for row in rows
        },
        build_intervention_for_row=_build_intervention,
        build_control_intervention_for_row=_build_intervention,
        generate_completions=generate,
        log_progress_every=log_progress_every,
    )


def test_generation_logs_progress_every_n_candidates(caplog) -> None:
    _build_generation_rows_for_logging(n_candidates=5, log_progress_every=2)
    progress = [
        r.message
        for r in caplog.records
        if "roundtrip generation:" in r.message
        and "candidates processed" in r.message
    ]
    # 5 candidates, logging every 2 -> progress lines at the 2nd and 4th.
    assert len(progress) == 2, progress
    assert "2/5 candidates processed" in progress[0]
    assert "4/5 candidates processed" in progress[1]


def test_generation_no_progress_logs_when_log_progress_every_zero(caplog) -> None:
    _build_generation_rows_for_logging(n_candidates=3, log_progress_every=0)
    assert not [
        r for r in caplog.records if "candidates processed" in r.message
    ]
