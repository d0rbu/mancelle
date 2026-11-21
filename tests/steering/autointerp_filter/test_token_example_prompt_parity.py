from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from steering.eval.autointerp_filter.parity import (
    compare_rendered_trial_surfaces,
    compare_trial_surfaces,
)
from steering.eval.autointerp_filter.prompts import (
    DEFAULT_INTRUDER_FEW_SHOT_SET,
    DEFAULT_INTRUDER_PROMPT_VARIANT,
)
from steering.eval.autointerp_filter.types import (
    ScoringTarget,
    TargetKind,
    TokenExampleSet,
    TokenWindowRef,
)


def _target() -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(0,),
        target_key="feature:0",
        display_name="feature_0",
    )


def _ref(
    doc_id: int,
    *,
    activations: tuple[float, ...],
    token_strings: tuple[str, ...] | None = None,
) -> TokenWindowRef:
    if token_strings is None:
        token_strings = (
            f"doc{doc_id} ",
            "café",
            " highlighted",
            " tail",
        )
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"source-{doc_id}",
        token_start=0,
        token_end=4,
        token_ids=(doc_id * 10, doc_id * 10 + 1, doc_id * 10 + 2, doc_id * 10 + 3),
        token_strings=token_strings,
        target_feature_ids=(0,),
        activations=activations,
        max_activation=max(activations),
        quantile=None,
        active=max(activations) > 0.0,
    )


def _examples(*, scale: float = 1.0) -> TokenExampleSet:
    positives = tuple(
        _ref(
            doc_id,
            activations=(0.0, scale * float(20 - doc_id), scale * 8.0, 0.0),
        )
        for doc_id in range(1, 9)
    )
    intruders = tuple(_ref(doc_id, activations=(0.0, 0.0, 0.0, 0.0)) for doc_id in range(101, 109))
    return TokenExampleSet(positives=positives, non_activating=intruders)


def _compare(reference: TokenExampleSet, candidate: TokenExampleSet, **kwargs):
    return compare_rendered_trial_surfaces(
        reference,
        candidate,
        n_positive_examples=2,
        n_trials=12,
        n_quantiles=2,
        target=_target(),
        seed=17,
        example_ctx_len=4,
        activation_threshold_ratio=0.3,
        **kwargs,
    )


def test_small_activation_drift_can_change_trial_keys_while_prompt_bytes_stay_exact() -> None:
    reference = _examples(scale=1.0)
    candidate = _examples(scale=1.001)

    logical = compare_trial_surfaces(
        reference,
        candidate,
        n_positive_examples=2,
        n_trials=12,
        n_quantiles=2,
        target=_target(),
        seed=17,
        example_ctx_len=4,
        activation_threshold_ratio=0.3,
    )
    rendered = _compare(reference, candidate)

    assert logical.reference_count == logical.candidate_count == 12
    assert logical.same_prompt_surface_count == 12
    assert logical.same_trial_key_count == 0
    assert rendered.reference_count == rendered.candidate_count == 12
    assert rendered.exact_rendered_examples_count == 12
    assert rendered.exact_message_bytes_count == 12
    assert rendered.all_rendered_examples_exact
    assert rendered.all_message_bytes_exact
    assert rendered.reference_messages_sha256 == rendered.candidate_messages_sha256


def test_rendered_prompt_report_to_json_exposes_boolean_acceptance_surface() -> None:
    report = _compare(_examples(), _examples())
    payload = report.to_json()
    assert payload == {
        "reference_count": 12,
        "candidate_count": 12,
        "exact_rendered_examples_count": 12,
        "exact_message_bytes_count": 12,
        "reference_messages_sha256": report.reference_messages_sha256,
        "candidate_messages_sha256": report.candidate_messages_sha256,
        "all_rendered_examples_exact": True,
        "all_message_bytes_exact": True,
    }


def test_token_string_drift_is_detected_in_rendered_examples_and_full_messages() -> None:
    reference = _examples()
    changed_first = replace(
        reference.positives[0],
        token_strings=("changed ", "café", " highlighted", " tail"),
    )
    candidate = replace(reference, positives=(changed_first, *reference.positives[1:]))

    report = _compare(reference, candidate)

    assert report.exact_rendered_examples_count < report.reference_count
    assert report.exact_message_bytes_count < report.reference_count
    assert not report.all_rendered_examples_exact
    assert not report.all_message_bytes_exact
    assert report.reference_messages_sha256 != report.candidate_messages_sha256


def test_threshold_crossing_is_detected_even_when_windows_and_tokens_are_identical() -> None:
    reference = _examples()
    first = reference.positives[0]
    # max=19: token 2 is highlighted at 8.0 in the reference (>=5.7), but not
    # at 5.0 in the candidate. Window identity remains otherwise unchanged.
    changed = replace(
        first,
        activations=(0.0, 19.0, 5.0, 0.0),
        max_activation=19.0,
    )
    candidate = replace(reference, positives=(changed, *reference.positives[1:]))

    report = _compare(reference, candidate)

    assert report.exact_rendered_examples_count < 12
    assert report.exact_message_bytes_count < 12
    assert not report.all_message_bytes_exact


@pytest.mark.parametrize("cot", [False, True])
@pytest.mark.parametrize(
    ("prompt_variant", "few_shot_set"),
    [
        (DEFAULT_INTRUDER_PROMPT_VARIANT, DEFAULT_INTRUDER_FEW_SHOT_SET),
        ("delphi", "delphi"),
        ("calibrated_v1", "sae_balanced_v1"),
    ],
)
def test_exact_prompt_comparison_covers_production_prompt_variants_and_cot(
    cot: bool,
    prompt_variant: str,
    few_shot_set: str,
) -> None:
    report = _compare(
        _examples(),
        _examples(),
        cot=cot,
        prompt_variant=prompt_variant,
        few_shot_set=few_shot_set,
    )
    assert report.all_rendered_examples_exact
    assert report.all_message_bytes_exact


def test_non_ascii_and_prompt_marker_escaping_are_hashed_as_exact_utf8_bytes() -> None:
    examples = _examples()
    first = replace(
        examples.positives[0],
        token_strings=("café <<literal>> ", "λ", " highlighted", " tail"),
    )
    examples = replace(examples, positives=(first, *examples.positives[1:]))
    report = _compare(examples, examples)
    assert report.all_message_bytes_exact
    assert len(bytes.fromhex(report.reference_messages_sha256)) == hashlib.sha256().digest_size


def test_empty_trial_surface_has_exact_empty_stream_digest() -> None:
    empty = TokenExampleSet(positives=(), non_activating=())
    report = _compare(empty, empty)
    assert report.reference_count == report.candidate_count == 0
    assert report.all_rendered_examples_exact
    assert report.all_message_bytes_exact
    assert report.reference_messages_sha256 == hashlib.sha256(b"").hexdigest()
