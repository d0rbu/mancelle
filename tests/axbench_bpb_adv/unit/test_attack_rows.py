from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pyarrow as pa
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.types import BpbScoreRow, SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackSample,
    BpbAttackWorkSpec,
    core_prefix_id,
    pair_samples_into_prefixes,
    prepare_bpb_attack_rows,
    work_specs_from_method_registry,
)
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GepaConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.optimize import build_control_candidate
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA, CANDIDATES_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table


def _attack_sample(
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
    content_class: Literal["concept_rich", "concept_avoiding"] = "concept_rich",
    prefix_text: str = "prefix",
    prefix_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
) -> BpbAttackSample:
    prefix = prefix_tokens if prefix_tokens is not None else [0, 10, 11, 12, 13]
    output = output_tokens if output_tokens is not None else [21, 22]
    return BpbAttackSample(
        sample_id=make_sample_id(
            concept_id=concept_id,
            mode=mode,
            content_class=content_class,
            prefix_text=prefix_text,
            output_text="|".join(str(token) for token in output),
        ),
        concept_id=concept_id,
        mode=mode,
        content_class=content_class,
        prefix_text=prefix_text,
        prefix_hash=make_prefix_hash(prefix_text),
        prefix_tokens=prefix,
        output_tokens=output,
    )


def _rich_avoid_pair(
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
    prefix_text: str = "prefix",
    prefix_tokens: list[int] | None = None,
    rich_output: list[int] | None = None,
    avoid_output: list[int] | None = None,
) -> list[BpbAttackSample]:
    prefix = prefix_tokens if prefix_tokens is not None else [0, 10, 11, 12, 13]
    return [
        _attack_sample(
            concept_id=concept_id,
            mode=mode,
            content_class="concept_rich",
            prefix_text=prefix_text,
            prefix_tokens=prefix,
            output_tokens=rich_output if rich_output is not None else [21, 22],
        ),
        _attack_sample(
            concept_id=concept_id,
            mode=mode,
            content_class="concept_avoiding",
            prefix_text=prefix_text,
            prefix_tokens=prefix,
            output_tokens=avoid_output if avoid_output is not None else [31, 32, 33],
        ),
    ]


def _multi_pair_samples(
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
    prefix_text: str = "prefix",
    prefix_tokens: list[int] | None = None,
    rich_outputs: list[list[int]],
    avoid_outputs: list[list[int]],
) -> list[BpbAttackSample]:
    """N rich + M avoid samples sharing one (concept, mode, prefix) group."""

    prefix = prefix_tokens if prefix_tokens is not None else [0, 10, 11, 12, 13]
    samples: list[BpbAttackSample] = []
    for rich_output in rich_outputs:
        samples.append(
            _attack_sample(
                concept_id=concept_id,
                mode=mode,
                content_class="concept_rich",
                prefix_text=prefix_text,
                prefix_tokens=prefix,
                output_tokens=rich_output,
            )
        )
    for avoid_output in avoid_outputs:
        samples.append(
            _attack_sample(
                concept_id=concept_id,
                mode=mode,
                content_class="concept_avoiding",
                prefix_text=prefix_text,
                prefix_tokens=prefix,
                output_tokens=avoid_output,
            )
        )
    return samples


def _prefix(
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
    prefix_text: str = "prefix",
    prefix_tokens: list[int] | None = None,
    rich_output: list[int] | None = None,
    avoid_output: list[int] | None = None,
) -> BpbAttackPrefix:
    (prefix,) = pair_samples_into_prefixes(
        _rich_avoid_pair(
            concept_id=concept_id,
            mode=mode,
            prefix_text=prefix_text,
            prefix_tokens=prefix_tokens,
            rich_output=rich_output,
            avoid_output=avoid_output,
        )
    )
    return prefix


def _spec(mode: SteeringMode = SteeringMode.POSITIVE) -> BpbAttackWorkSpec:
    return BpbAttackWorkSpec(
        method_name="diffmean_encdec",
        mode=mode,
        target_model="encdec",
        coefficient_str="1.0",
        baseline_family="diffmean",
    )


def _target_meta(*, negative_selection=None) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=2,
        n_sae_features=8,
        negative_selection=negative_selection,
        feature_selection=None,
        feature_mask=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=256,
    )


# --- 2.3 pairing -----------------------------------------------------------


def test_pair_samples_groups_rich_and_avoid_into_one_prefix() -> None:
    prefixes = pair_samples_into_prefixes(_rich_avoid_pair())
    assert len(prefixes) == 1
    prefix = prefixes[0]
    assert prefix.concept_id == 1
    assert prefix.mode == SteeringMode.POSITIVE
    assert prefix.prefix_text == "prefix"
    assert prefix.prefix_tokens == [0, 10, 11, 12, 13]
    assert prefix.concept_rich_output_tokens == [[21, 22]]
    assert prefix.concept_avoiding_output_tokens == [[31, 32, 33]]
    assert len(prefix.concept_rich_sample_ids) == 1
    assert len(prefix.concept_avoiding_sample_ids) == 1


def test_core_prefix_id_stable_across_the_two_completions() -> None:
    samples = _rich_avoid_pair()
    rich, avoid = samples
    assert rich.sample_id != avoid.sample_id
    expected = core_prefix_id(
        concept_id=1, mode=SteeringMode.POSITIVE, prefix_text="prefix"
    )
    prefix = pair_samples_into_prefixes(samples)[0]
    assert prefix.core_prefix_id == expected
    # Independent of which completion order they arrive in.
    reversed_prefix = pair_samples_into_prefixes([avoid, rich])[0]
    assert reversed_prefix.core_prefix_id == expected
    assert len(expected) == 16


def test_core_prefix_id_distinguishes_concept_mode_and_prefix() -> None:
    base = core_prefix_id(
        concept_id=1, mode=SteeringMode.POSITIVE, prefix_text="prefix"
    )
    assert base != core_prefix_id(
        concept_id=2, mode=SteeringMode.POSITIVE, prefix_text="prefix"
    )
    assert base != core_prefix_id(
        concept_id=1, mode=SteeringMode.NEGATIVE, prefix_text="prefix"
    )
    assert base != core_prefix_id(
        concept_id=1, mode=SteeringMode.POSITIVE, prefix_text="other"
    )


def test_pair_samples_errors_on_missing_avoid_side() -> None:
    rich, _avoid = _rich_avoid_pair()
    with pytest.raises(AxBenchBpbAdvError, match="concept_avoiding"):
        pair_samples_into_prefixes([rich])


def test_pair_samples_errors_on_missing_rich_side() -> None:
    _rich, avoid = _rich_avoid_pair()
    with pytest.raises(AxBenchBpbAdvError, match="concept_rich"):
        pair_samples_into_prefixes([avoid])


def test_pair_samples_accepts_multiple_rich_completions_as_a_list() -> None:
    # Gap B: a second rich completion is no longer a "duplicate" error; it is a
    # legitimate extra completion carried alongside the first.
    rich, avoid = _rich_avoid_pair()
    second_rich = _attack_sample(
        content_class="concept_rich",
        prefix_text="prefix",
        output_tokens=[41, 42],
    )
    prefixes = pair_samples_into_prefixes([rich, avoid, second_rich])
    assert len(prefixes) == 1
    prefix = prefixes[0]
    assert prefix.concept_rich_output_tokens == [[21, 22], [41, 42]]
    assert prefix.concept_avoiding_output_tokens == [[31, 32, 33]]
    assert len(prefix.concept_rich_sample_ids) == 2


def test_pair_samples_errors_even_when_other_groups_are_complete() -> None:
    complete = _rich_avoid_pair(prefix_text="complete")
    rich_only, _ = _rich_avoid_pair(prefix_text="incomplete")
    with pytest.raises(AxBenchBpbAdvError, match="incomplete"):
        pair_samples_into_prefixes([*complete, rich_only])


def test_pair_samples_asserts_both_sides_share_prefix_tokens() -> None:
    rich = _attack_sample(
        content_class="concept_rich",
        prefix_text="prefix",
        prefix_tokens=[0, 10, 11, 12, 13],
    )
    avoid = _attack_sample(
        content_class="concept_avoiding",
        prefix_text="prefix",
        prefix_tokens=[0, 10, 11, 12, 99],
    )
    with pytest.raises(AssertionError):
        pair_samples_into_prefixes([rich, avoid])


# --- 2.2/2.3 dual-completion attack rows -----------------------------------


def _cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )


def test_attack_rows_carry_dual_completion_fields_and_digests() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(0, 2))
    ]

    table = prepare_bpb_attack_rows(
        prefixes=[_prefix()], work_specs=[_spec()], cfg=cfg
    )

    assert table.schema.equals(ATTACK_ROWS_SCHEMA)
    rows = table.to_pylist()
    assert len(rows) == 2
    prefix = _prefix()
    for row in rows:
        assert "content_class" not in row
        assert "output_tokens" not in row
        assert row["core_prefix_id"] == prefix.core_prefix_id
        assert row["concept_rich_sample_id"] == prefix.concept_rich_sample_ids[0]
        assert (
            row["concept_avoiding_sample_id"]
            == prefix.concept_avoiding_sample_ids[0]
        )
        assert row["concept_rich_output_tokens"] == [21, 22]
        assert row["concept_avoiding_output_tokens"] == [31, 32, 33]
        assert row["concept_rich_output_digest"] == token_sequence_digest([21, 22])
        assert row["concept_avoiding_output_digest"] == token_sequence_digest(
            [31, 32, 33]
        )


def test_attack_row_id_keyed_on_core_prefix_id_and_both_output_digests() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(0, 2))
    ]

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()], work_specs=[_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 2
    assert len({row["attack_row_id"] for row in rows}) == len(rows)

    # Differing rich output (same core_prefix_id) yields different attack_row_ids.
    other = prepare_bpb_attack_rows(
        prefixes=[_prefix(rich_output=[99, 98])], work_specs=[_spec()], cfg=cfg
    ).to_pylist()
    base_ids = {row["attack_row_id"] for row in rows}
    other_ids = {row["attack_row_id"] for row in other}
    assert base_ids.isdisjoint(other_ids)


def test_attack_rows_encode_strength_context_and_final_k_span() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(0, 2))
    ]

    table = prepare_bpb_attack_rows(
        prefixes=[_prefix()], work_specs=[_spec()], cfg=cfg
    )

    rows = table.to_pylist()
    assert len(rows) == 2
    control = next(row for row in rows if row["adversarial_strength"] == 0.0)
    edited = next(row for row in rows if row["adversarial_strength"] == 2.0)
    assert control["editable_token_start"] == 5
    assert control["editable_token_count"] == 0
    assert control["original_editable_tokens"] == []
    assert edited["editable_token_start"] == 3
    assert edited["editable_token_count"] == 2
    assert edited["original_editable_tokens"] == [12, 13]
    assert {row["context_signal_mode"] for row in rows} == {"fixed_context"}
    assert len({row["attack_row_id"] for row in rows}) == len(rows)


def test_prompt_token_rows_are_fixed_context_only_emitted_once() -> None:
    # Same run emits BOTH prompt-token (faster_gcg) rows and additive_direction
    # rows so the {faster_gcg, gepa} filter is load-bearing: it must isolate the
    # fixed_context rows from the not_applicable additive rows. The decoder_only +
    # diffmean work spec satisfies _spec_supports_additive (diffmean is in the
    # default additive families), mirroring the additive coexistence fixture.
    cfg = _cfg()
    cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(0, 2)),
        AdditiveDirectionConfig(magnitudes=(0.0, 2.0)),
    ]
    prefixes = [_prefix()]
    work_specs = [
        BpbAttackWorkSpec(
            method_name="diffmean_decoder_only",
            mode=SteeringMode.POSITIVE,
            target_model="decoder_only",
            coefficient_str="1.0",
            baseline_family="diffmean",
        )
    ]

    table = prepare_bpb_attack_rows(prefixes=prefixes, work_specs=work_specs, cfg=cfg)
    rows = table.to_pylist()

    prompt_rows = [r for r in rows if r["adversarial_method"] in {"faster_gcg", "gepa"}]
    assert prompt_rows, "expected prompt-token rows"
    assert {r["context_signal_mode"] for r in prompt_rows} == {"fixed_context"}

    additive_rows = [
        r for r in rows if r["adversarial_method"] == "additive_direction"
    ]
    assert additive_rows, "expected additive_direction rows in the same run"
    assert {r["context_signal_mode"] for r in additive_rows} == {"not_applicable"}

    ids = [r["attack_row_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_attack_row_identity_changes_with_score_affecting_fields() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(1,)),
        GepaConfig(
            metric_call_budgets=(8,),
            k_editable_tokens=1,
            reflection_model="test-reflector",
        ),
    ]
    rows = prepare_bpb_attack_rows(
        prefixes=[
            _prefix(prefix_text="prefix-a"),
            _prefix(prefix_text="prefix-b", prefix_tokens=[0, 10, 11, 12, 99]),
        ],
        work_specs=[
            _spec(),
            BpbAttackWorkSpec(
                method_name="diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="2.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="diffmean_decoder",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ],
        cfg=cfg,
    ).to_pylist()

    assert len({row["attack_row_id"] for row in rows}) == len(rows)


def test_target_only_work_spec_emits_only_prompt_control_row() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(0, 1, 2))]
    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="feature_diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="None",
                baseline_family="ours",
                target_only=True,
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert len(rows) == 1
    row = rows[0]
    assert row["method_name"] == "feature_diffmean_encdec"
    assert row["adversarial_method"] == "faster_gcg"
    assert row["adversarial_strength"] == 0.0
    assert row["strength_index"] == 0
    assert row["editable_token_count"] == 0


def test_target_only_work_spec_requires_explicit_prompt_control_row() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(1, 2))]

    with pytest.raises(AxBenchBpbAdvError, match="explicit no-adversary"):
        prepare_bpb_attack_rows(
            prefixes=[_prefix()],
            work_specs=[
                BpbAttackWorkSpec(
                    method_name="feature_diffmean_encdec",
                    mode=SteeringMode.POSITIVE,
                    target_model="encdec",
                    coefficient_str="None",
                    baseline_family="ours",
                    target_only=True,
                )
            ],
            cfg=cfg,
        )


def test_target_only_gepa_work_spec_synthesizes_control_row() -> None:
    # GEPA's editable-token count is config-validated >= 1, so its grid can never
    # supply a no-adversary control the way faster_gcg (k=0) can. A target-only
    # feature-DiffMean method still needs its no-adversary reference under GEPA,
    # so exactly one control row is SYNTHESIZED rather than aborting the run.
    cfg = _cfg()
    cfg.adversarial_methods = [
        GepaConfig(metric_call_budgets=(8, 16), k_editable_tokens=2, reflection_model="x")
    ]

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="feature_diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="None",
                baseline_family="ours",
                target_only=True,
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert len(rows) == 1
    row = rows[0]
    assert row["method_name"] == "feature_diffmean_encdec"
    assert row["adversarial_method"] == "gepa"
    assert row["adversarial_strength"] == 0.0
    assert row["strength_index"] == 0
    assert row["editable_token_count"] == 0


def test_additive_direction_uses_not_applicable_context_once_per_magnitude() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[AdditiveDirectionConfig(magnitudes=(0.0, 2.0))],
    )

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert len(rows) == 2
    assert {row["context_signal_mode"] for row in rows} == {"not_applicable"}
    assert [row["adversarial_strength"] for row in rows] == [0.0, 2.0]


def test_fair_feature_diffmean_base_method_emits_additive_control_target() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("diffmean",),
                magnitudes=(0.0, 2.0),
            )
        ],
    )

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="feature_diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="None",
                baseline_family="ours",
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert len(rows) == 1
    row = rows[0]
    assert row["method_name"] == "feature_diffmean_encdec"
    assert row["adversarial_method"] == "additive_direction"
    assert row["adversarial_strength"] == 0.0
    assert row["strength_unit"] == "counter_direction_magnitude"
    assert row["context_signal_mode"] == "not_applicable"


def test_fair_feature_diffmean_keeps_supported_additive_grid_and_control() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("ours",),
                magnitudes=(0.0, 0.2, 2.0),
            )
        ],
    )

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="feature_diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="None",
                baseline_family="ours",
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert [row["adversarial_strength"] for row in rows] == [0.0, 0.2, 2.0]
    assert {row["adversarial_method"] for row in rows} == {"additive_direction"}
    assert {row["context_signal_mode"] for row in rows} == {"not_applicable"}


def test_fair_feature_diffmean_additive_grid_requires_explicit_control() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            AdditiveDirectionConfig(
                families=("ours",),
                magnitudes=(0.2, 2.0),
            )
        ],
    )

    with pytest.raises(AxBenchBpbAdvError, match="explicit no-adversary"):
        prepare_bpb_attack_rows(
            prefixes=[_prefix()],
            work_specs=[
                BpbAttackWorkSpec(
                    method_name="feature_diffmean_encdec",
                    mode=SteeringMode.POSITIVE,
                    target_model="encdec",
                    coefficient_str="None",
                    baseline_family="ours",
                )
            ],
            cfg=cfg,
        )


def test_fair_feature_diffmean_gepa_grid_synthesizes_control() -> None:
    # An ACTIVE feature-DiffMean base method is attacked at every GEPA budget AND
    # keeps a synthesized no-adversary control (strength 0) so representative
    # selection has its target under GEPA. The control sorts first by
    # strength_index; the real budgets shift up one ordinal so strength_index
    # stays a unique, strength-ordered key per series.
    cfg = _cfg()
    cfg.adversarial_methods = [
        GepaConfig(metric_call_budgets=(8, 16), k_editable_tokens=2, reflection_model="x")
    ]

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="feature_diffmean_encdec",
                mode=SteeringMode.POSITIVE,
                target_model="encdec",
                coefficient_str="None",
                baseline_family="ours",
            )
        ],
        cfg=cfg,
    ).to_pylist()

    assert {row["adversarial_method"] for row in rows} == {"gepa"}
    assert [row["adversarial_strength"] for row in rows] == [0.0, 8.0, 16.0]
    assert [row["strength_index"] for row in rows] == [0, 1, 2]
    assert [row["editable_token_count"] for row in rows] == [0, 2, 2]
    controls = [row for row in rows if row["adversarial_strength"] == 0.0]
    assert len(controls) == 1
    assert controls[0]["editable_token_count"] == 0


def test_faster_gcg_emits_control_row_for_lora_but_additive_skips_it() -> None:
    """LoRA appears on the plots only via the method-agnostic faster_gcg control.

    ``additive_direction`` is a residual counter-pressure on top of a residual
    steering direction, so ``_spec_supports_additive`` restricts it to the
    diffmean / linear_probe families -- LoRA (``baseline_family='lora'``) is
    skipped, which is why LoRA was absent from the plots. faster_gcg is a
    prompt-token attack that applies to EVERY work spec, so its ``k=0`` control
    row (prompt unmodified) yields one LoRA control point per coefficient on the
    steerability-vs-baseline scatter.
    """

    lora_spec = BpbAttackWorkSpec(
        method_name="lora_decoder_only",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
        coefficient_str="1.0",
        baseline_family="lora",
    )

    additive_cfg = _cfg()
    additive_cfg.adversarial_methods = [AdditiveDirectionConfig(magnitudes=(0.0, 2.0))]
    additive_rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()], work_specs=[lora_spec], cfg=additive_cfg
    ).to_pylist()
    assert additive_rows == []  # family filter excludes LoRA from additive

    gcg_cfg = _cfg()
    gcg_cfg.adversarial_methods = [
        FasterGcgConfig(optimized_token_counts=(0,))
    ]
    gcg_rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()], work_specs=[lora_spec], cfg=gcg_cfg
    ).to_pylist()
    assert len(gcg_rows) == 1
    (row,) = gcg_rows
    assert row["method_name"] == "lora_decoder_only"
    assert row["adversarial_method"] == "faster_gcg"
    assert row["adversarial_strength"] == 0.0
    assert row["strength_index"] == 0
    assert row["editable_token_count"] == 0  # k=0 control leaves the prompt intact


def test_spec_supports_additive_for_lora_and_encdec_ours_in_families() -> None:
    """additive now composes a counter-direction on top of ANY baseline, so
    _spec_supports_additive accepts lora and encdec `ours` specs (no longer
    decoder_only-restricted) when their family is in additive_cfg.families."""
    from steering.eval.axbench_bpb_adv.attack_rows import _spec_supports_additive

    additive_cfg = AdditiveDirectionConfig(
        families=("diffmean", "lora", "ours")
    )

    def _spec(method_name, target_model, baseline_family):
        return BpbAttackWorkSpec(
            method_name=method_name,
            mode=SteeringMode.POSITIVE,
            target_model=target_model,
            coefficient_str="None",
            baseline_family=baseline_family,
        )

    assert _spec_supports_additive(
        _spec("lora_decoder_only", "decoder_only", "lora"), additive_cfg
    )
    # encdec target is now allowed (the decoder_only-only gate is gone).
    assert _spec_supports_additive(
        _spec("ours_replace_all_encdec", "encdec", "ours"), additive_cfg
    )
    assert _spec_supports_additive(
        _spec("diffmean_decoder_only", "decoder_only", "diffmean"), additive_cfg
    )
    # linear_probe not listed in families -> excluded.
    assert not _spec_supports_additive(
        _spec("linear_probe_decoder_only", "decoder_only", "linear_probe"),
        additive_cfg,
    )


def test_spec_supports_additive_always_includes_control_baseline() -> None:
    """Control baselines are counter-steered whenever additive is configured,
    regardless of which steering families the user chose to attack."""
    from steering.eval.axbench_bpb_adv.attack_rows import _spec_supports_additive

    additive_cfg = AdditiveDirectionConfig(families=("lora",))  # control NOT listed

    def _spec(name, target):
        return BpbAttackWorkSpec(
            method_name=name, mode=SteeringMode.POSITIVE,
            target_model=target, coefficient_str="None", baseline_family="control",
        )

    assert _spec_supports_additive(_spec("control_decoder_only", "decoder_only"), additive_cfg)
    assert _spec_supports_additive(_spec("control_encdec", "encdec"), additive_cfg)
    # A non-control baseline NOT in families is still excluded.
    diffmean_spec = BpbAttackWorkSpec(
        method_name="diffmean_decoder_only", mode=SteeringMode.POSITIVE,
        target_model="decoder_only", coefficient_str="1.0", baseline_family="diffmean",
    )
    assert not _spec_supports_additive(diffmean_spec, additive_cfg)


def test_work_specs_include_control_methods_for_each_target() -> None:
    """The 4 control methods flow through work_specs_from_method_registry as
    normal registry methods (one decoder_only + one encdec, both modes)."""
    cfg = _cfg()
    specs = work_specs_from_method_registry(
        base_methods=None,
        target_models=["decoder_only", "encdec"],
        target_model_metas={"decoder_only": _target_meta(), "encdec": _target_meta()},
        cfg=cfg,
    )
    control_names = {s.method_name for s in specs if s.baseline_family == "control"}
    assert control_names == {
        "control_decoder_only",
        "neg_control_decoder_only",
        "control_encdec",
        "neg_control_encdec",
    }


def test_prompt_strength_larger_than_prompt_fails_instead_of_skipping_sample() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            FasterGcgConfig(optimized_token_counts=(2,))
        ],
    )

    with pytest.raises(AxBenchBpbAdvError, match="fewer than k=2"):
        prepare_bpb_attack_rows(
            prefixes=[
                _prefix(prefix_text="short", prefix_tokens=[0, 10]),
                _prefix(prefix_text="long", prefix_tokens=[0, 10, 11]),
            ],
            work_specs=[_spec()],
            cfg=cfg,
        )


def test_base_method_selection_rejects_target_mismatches_and_unsupported_controls() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        base_methods=["diffmean_decoder_only"],
    )

    with pytest.raises(AxBenchBpbAdvError, match="target models"):
        work_specs_from_method_registry(
            base_methods=cfg.base_methods,
            target_models=["encdec"],
            target_model_metas={"encdec": _target_meta()},
            cfg=cfg,
        )


def test_base_method_selection_rejects_removed_etc_as_unknown() -> None:
    """The removed extract-then-condition method names are no longer in the
    registry, so requesting one as a base method fails loudly as unknown
    (rather than the old "does not support" branch)."""
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        base_methods=["ours_extract_then_condition_encdec"],
    )
    with pytest.raises(AxBenchBpbAdvError, match="unknown BPB-Adv base method"):
        work_specs_from_method_registry(
            base_methods=cfg.base_methods,
            target_models=["encdec"],
            target_model_metas={"encdec": _target_meta()},
            cfg=cfg,
        )


def test_work_specs_from_registry_marks_representative_targets_target_only() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        base_methods=["diffmean_decoder_only"],
        representative_target_methods=["feature_diffmean_encdec"],
    )

    specs = work_specs_from_method_registry(
        base_methods=cfg.base_methods,
        target_models=["decoder_only", "encdec"],
        target_model_metas={
            "decoder_only": _target_meta(),
            "encdec": _target_meta(),
        },
        cfg=cfg,
    )

    by_name: Mapping[str, BpbAttackWorkSpec] = {
        spec.method_name: spec for spec in specs
    }
    assert by_name["diffmean_decoder_only"].target_only is False
    assert by_name["feature_diffmean_encdec"].target_only is True


def test_work_specs_from_registry_deduplicates_all_method_representative_targets() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        base_methods=None,
        representative_target_methods=["feature_diffmean_encdec"],
    )

    specs = work_specs_from_method_registry(
        base_methods=cfg.base_methods,
        target_models=["encdec"],
        target_model_metas={"encdec": _target_meta()},
        cfg=cfg,
    )

    feature_diffmean_specs = [
        spec for spec in specs if spec.method_name == "feature_diffmean_encdec"
    ]
    assert len(feature_diffmean_specs) == 1
    assert feature_diffmean_specs[0].target_only is False


def test_all_method_representative_targets_do_not_duplicate_candidate_ids() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        base_methods=None,
        representative_target_methods=["feature_diffmean_encdec"],
        adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0,))],
    )
    specs = work_specs_from_method_registry(
        base_methods=cfg.base_methods,
        target_models=["encdec"],
        target_model_metas={"encdec": _target_meta()},
        cfg=cfg,
    )
    feature_diffmean_specs = [
        spec for spec in specs if spec.method_name == "feature_diffmean_encdec"
    ]

    rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=feature_diffmean_specs,
        cfg=cfg,
    ).to_pylist()
    candidate_rows = [
        build_control_candidate(row=row, tokenizer=_TinyTokenizer()) for row in rows
    ]

    assert len(rows) == 1
    assert len({row["attack_row_id"] for row in rows}) == 1
    assert len({row["candidate_id"] for row in candidate_rows}) == 1


# --- error branches: work-spec / prepare entry points -----------------------


def test_work_specs_from_registry_rejects_empty_target_models() -> None:
    cfg = _cfg()
    with pytest.raises(
        AxBenchBpbAdvError, match="target_models must contain at least one target"
    ):
        work_specs_from_method_registry(
            base_methods=None,
            target_models=[],
            target_model_metas={},
            cfg=cfg,
        )


def test_work_specs_from_registry_rejects_unknown_target_metadata() -> None:
    cfg = _cfg()
    with pytest.raises(
        AxBenchBpbAdvError, match="missing target model metadata for \\['encdec'\\]"
    ):
        work_specs_from_method_registry(
            base_methods=None,
            target_models=["encdec"],
            target_model_metas={},
            cfg=cfg,
        )


def test_work_specs_from_registry_rejects_empty_resulting_grid() -> None:
    cfg = _cfg()
    # An explicit empty enable filter keeps no methods, so no method/coefficient
    # work specs survive -> the empty-grid guard fires.
    with pytest.raises(
        AxBenchBpbAdvError,
        match="no BPB-Adv method/coefficient specs survived",
    ):
        work_specs_from_method_registry(
            base_methods=[],
            target_models=["encdec"],
            target_model_metas={"encdec": _target_meta()},
            cfg=cfg,
        )


def test_prepare_bpb_attack_rows_rejects_empty_prefixes() -> None:
    cfg = _cfg()
    with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
        prepare_bpb_attack_rows(prefixes=[], work_specs=[_spec()], cfg=cfg)


def test_prepare_bpb_attack_rows_rejects_empty_work_specs() -> None:
    cfg = _cfg()
    with pytest.raises(BeartypeCallHintParamViolation, match="work_specs"):
        prepare_bpb_attack_rows(prefixes=[_prefix()], work_specs=[], cfg=cfg)


# --- 2B Gap B: cartesian rich x avoid pairs + pair_subsample ---------------


def _additive_cfg(*, pair_subsample: int = 0) -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[AdditiveDirectionConfig(magnitudes=(0.0,))],
        pair_subsample=pair_subsample,
    )
    return cfg


def _additive_spec() -> BpbAttackWorkSpec:
    return BpbAttackWorkSpec(
        method_name="diffmean_decoder_only",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
        coefficient_str="1.0",
        baseline_family="diffmean",
    )


def test_pair_samples_keeps_all_rich_and_avoid_completions_as_lists() -> None:
    samples = _multi_pair_samples(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    prefixes = pair_samples_into_prefixes(samples)
    assert len(prefixes) == 1
    prefix = prefixes[0]
    # All three rich and both avoid completions are carried as lists.
    assert prefix.concept_rich_output_tokens == [[21, 22], [23, 24], [25, 26]]
    assert prefix.concept_avoiding_output_tokens == [[31, 32], [33, 34]]
    assert len(prefix.concept_rich_sample_ids) == 3
    assert len(prefix.concept_avoiding_sample_ids) == 2
    # The shared prefix tokens are preserved exactly once.
    assert prefix.prefix_tokens == [0, 10, 11, 12, 13]


def test_pair_samples_still_errors_on_zero_rich_side() -> None:
    samples = _multi_pair_samples(
        rich_outputs=[],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    with pytest.raises(AxBenchBpbAdvError, match="concept_rich"):
        pair_samples_into_prefixes(samples)


def test_pair_samples_still_errors_on_zero_avoid_side() -> None:
    samples = _multi_pair_samples(
        rich_outputs=[[21, 22], [23, 24]],
        avoid_outputs=[],
    )
    with pytest.raises(AxBenchBpbAdvError, match="concept_avoiding"):
        pair_samples_into_prefixes(samples)


def test_pair_samples_asserts_shared_prefix_tokens_across_all_completions() -> None:
    samples = _multi_pair_samples(
        rich_outputs=[[21, 22], [23, 24]],
        avoid_outputs=[[31, 32]],
    )
    # Corrupt the second rich sample's prefix tokens.
    bad_rich = _attack_sample(
        content_class="concept_rich",
        prefix_text="prefix",
        prefix_tokens=[0, 10, 11, 12, 99],
        output_tokens=[27, 28],
    )
    with pytest.raises(AssertionError):
        pair_samples_into_prefixes([*samples, bad_rich])


def _multi_pair_prefix(
    *,
    rich_outputs: list[list[int]],
    avoid_outputs: list[list[int]],
    prefix_text: str = "prefix",
) -> BpbAttackPrefix:
    (prefix,) = pair_samples_into_prefixes(
        _multi_pair_samples(
            prefix_text=prefix_text,
            rich_outputs=rich_outputs,
            avoid_outputs=avoid_outputs,
        )
    )
    return prefix


def test_cartesian_explosion_emits_one_row_per_rich_avoid_pair() -> None:
    cfg = _additive_cfg(pair_subsample=0)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    # 3 rich x 2 avoid x 1 magnitude x 1 context = 6 attack rows.
    assert len(rows) == 6
    # Each row carries exactly one rich and one avoid completion (lists not nested).
    rich_seen = {tuple(row["concept_rich_output_tokens"]) for row in rows}
    avoid_seen = {tuple(row["concept_avoiding_output_tokens"]) for row in rows}
    assert rich_seen == {(21, 22), (23, 24), (25, 26)}
    assert avoid_seen == {(31, 32), (33, 34)}
    # The full cartesian product appears exactly once each.
    pairs = {
        (tuple(row["concept_rich_output_tokens"]), tuple(row["concept_avoiding_output_tokens"]))
        for row in rows
    }
    assert len(pairs) == 6
    # All rows share the same core prefix and prefix tokens.
    assert {row["core_prefix_id"] for row in rows} == {prefix.core_prefix_id}
    assert {tuple(row["source_prompt_tokens"]) for row in rows} == {
        tuple(prefix.prefix_tokens)
    }
    # Distinct attack_row_ids per pair (keyed on both output digests).
    assert len({row["attack_row_id"] for row in rows}) == len(rows)


def test_cartesian_rows_pair_sample_ids_with_their_output_tokens() -> None:
    cfg = _additive_cfg(pair_subsample=0)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24]],
        avoid_outputs=[[31, 32], [33, 34], [35, 36]],
    )
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 6
    rich_id_by_output = dict(
        zip(
            (tuple(out) for out in prefix.concept_rich_output_tokens),
            prefix.concept_rich_sample_ids,
            strict=True,
        )
    )
    avoid_id_by_output = dict(
        zip(
            (tuple(out) for out in prefix.concept_avoiding_output_tokens),
            prefix.concept_avoiding_sample_ids,
            strict=True,
        )
    )
    for row in rows:
        rich_key = tuple(row["concept_rich_output_tokens"])
        avoid_key = tuple(row["concept_avoiding_output_tokens"])
        assert row["concept_rich_sample_id"] == rich_id_by_output[rich_key]
        assert row["concept_avoiding_sample_id"] == avoid_id_by_output[avoid_key]
        assert row["concept_rich_output_digest"] == token_sequence_digest(
            list(rich_key)
        )
        assert row["concept_avoiding_output_digest"] == token_sequence_digest(
            list(avoid_key)
        )


def test_pair_subsample_caps_pair_count_deterministically() -> None:
    cfg = _additive_cfg(pair_subsample=4)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    first = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    second = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    # 3 x 2 = 6 pairs > cap 4 -> exactly 4 rows kept.
    assert len(first) == 4
    assert len(second) == 4
    # Deterministic: run-twice yields the identical surviving rows.
    assert [row["attack_row_id"] for row in first] == [
        row["attack_row_id"] for row in second
    ]
    # The surviving rows are a subset of the full cartesian product.
    full = prepare_bpb_attack_rows(
        prefixes=[prefix],
        work_specs=[_additive_spec()],
        cfg=_additive_cfg(pair_subsample=0),
    ).to_pylist()
    full_ids = {row["attack_row_id"] for row in full}
    assert {row["attack_row_id"] for row in first}.issubset(full_ids)


def test_pair_subsample_zero_keeps_all_pairs() -> None:
    cfg = _additive_cfg(pair_subsample=0)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 6


def test_pair_subsample_above_count_keeps_all_pairs() -> None:
    cfg = _additive_cfg(pair_subsample=100)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
    )
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 6


def test_pair_subsample_seed_differs_across_distinct_core_prefixes() -> None:
    cfg = _additive_cfg(pair_subsample=4)
    prefix_a = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
        prefix_text="prefix-a",
    )
    prefix_b = _multi_pair_prefix(
        rich_outputs=[[21, 22], [23, 24], [25, 26]],
        avoid_outputs=[[31, 32], [33, 34]],
        prefix_text="prefix-b",
    )
    assert prefix_a.core_prefix_id != prefix_b.core_prefix_id
    rows_a = prepare_bpb_attack_rows(
        prefixes=[prefix_a], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    rows_b = prepare_bpb_attack_rows(
        prefixes=[prefix_b], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows_a) == len(rows_b) == 4
    # Different core_prefix_id -> independent (and generally different) subsample
    # of the (rich_output, avoid_output) pair selection.
    pairs_a = {
        (tuple(row["concept_rich_output_tokens"]), tuple(row["concept_avoiding_output_tokens"]))
        for row in rows_a
    }
    pairs_b = {
        (tuple(row["concept_rich_output_tokens"]), tuple(row["concept_avoiding_output_tokens"]))
        for row in rows_b
    }
    assert pairs_a != pairs_b


def test_single_pair_prefix_still_emits_one_pair_unchanged() -> None:
    cfg = _additive_cfg(pair_subsample=1)
    prefix = _multi_pair_prefix(
        rich_outputs=[[21, 22]],
        avoid_outputs=[[31, 32, 33]],
    )
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 1
    assert rows[0]["concept_rich_output_tokens"] == [21, 22]
    assert rows[0]["concept_avoiding_output_tokens"] == [31, 32, 33]


# --- BUG 2: duplicate-completion id collision --------------------------------


class _TinyTokenizer:
    """Minimal tokenizer covering decode/vocab needs of the scoring path."""

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 256

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _dual_score_pair(candidate_row, rich_input, avoid_input):
    def _row(scoring_input, bpb: float):
        return BpbScoreRow(
            method_name=str(candidate_row["method_name"]),
            concept_id=int(candidate_row["concept_id"]),
            mode=str(candidate_row["mode"]),
            content_class=scoring_input.content_class,
            target=str(candidate_row["target_model"]),
            coefficient_str=str(candidate_row["coefficient_str"]),
            sample_id=scoring_input.sample_id,
            bpb_intervened=bpb,
            n_output_tokens=scoring_input.n_output,
            n_output_bytes=2,
            truncated=False,
        )

    return _row(rich_input, 1.0), _row(avoid_input, 3.0)


def test_duplicate_rich_completion_collapses_to_one_pair_and_one_candidate_id() -> None:
    # A prefix carrying TWO byte-identical concept_rich completions plus one
    # avoid completion. Without dedup, _all_prefix_pairs would explode into two
    # pairs ((r0,a0) and (r1,a0)) that share the SAME (rich_digest, avoid_digest)
    # tuple -> colliding attack_row_ids and candidate_ids, which trips the
    # "duplicate candidate_id" guard in build_bpb_adv_score_table.
    cfg = _additive_cfg(pair_subsample=0)
    prefix = _multi_pair_prefix(
        rich_outputs=[[10, 11], [10, 11]],
        avoid_outputs=[[20, 21]],
    )
    # Dedup collapses the identical rich completions to one representative.
    assert prefix.concept_rich_output_tokens == [[10, 11]]
    assert len(prefix.concept_rich_sample_ids) == 1
    assert prefix.concept_avoiding_output_tokens == [[20, 21]]

    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    # Exactly one (rich x avoid) pair survives, with a single distinct id.
    assert len(rows) == 1
    assert len({row["attack_row_id"] for row in rows}) == 1

    # End-to-end: building candidates + the score table must NOT raise a
    # duplicate-candidate_id error.
    tokenizer = _TinyTokenizer()
    candidate_rows = [build_control_candidate(row=row, tokenizer=tokenizer) for row in rows]
    assert len({row["candidate_id"] for row in candidate_rows}) == len(candidate_rows)
    candidates = pa.Table.from_pylist(candidate_rows, schema=CANDIDATES_SCHEMA)

    scores = build_bpb_adv_score_table(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=tokenizer,
        token_bytes=torch.ones(256, dtype=torch.int64),
        max_seq_len_by_target={"decoder_only": 64},
        score_pair=_dual_score_pair,
    )
    assert scores.num_rows == 1


def test_distinct_rich_completions_still_yield_distinct_candidate_ids() -> None:
    # Regression guard: dedup must NOT collapse genuinely distinct completions.
    cfg = _additive_cfg(pair_subsample=0)
    prefix = _multi_pair_prefix(
        rich_outputs=[[10, 11], [12, 13]],
        avoid_outputs=[[20, 21]],
    )
    assert prefix.concept_rich_output_tokens == [[10, 11], [12, 13]]
    rows = prepare_bpb_attack_rows(
        prefixes=[prefix], work_specs=[_additive_spec()], cfg=cfg
    ).to_pylist()
    assert len(rows) == 2

    tokenizer = _TinyTokenizer()
    candidate_rows = [build_control_candidate(row=row, tokenizer=tokenizer) for row in rows]
    assert len({row["candidate_id"] for row in candidate_rows}) == 2
    candidates = pa.Table.from_pylist(candidate_rows, schema=CANDIDATES_SCHEMA)
    scores = build_bpb_adv_score_table(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=tokenizer,
        token_bytes=torch.ones(256, dtype=torch.int64),
        max_seq_len_by_target={"decoder_only": 64},
        score_pair=_dual_score_pair,
    )
    assert scores.num_rows == 2


# ---------------------------------------------------------------------------
# #1249 follow-up: exclude_concept_modes
# ---------------------------------------------------------------------------
#
# The one legitimate use: a (concept, mode) whose direction the #1228 scale
# guard refuses (concept 37 positive on the l22 clustered source: every fit
# projection negative, max_activation -1098), where counter-steering is
# undefined. Exclusion must be LOUD -- silent row drops are this pipeline's
# most-repeated failure class (the roofline arm lost 693 bags to one).


def test_excluded_concept_mode_emits_no_rows_for_that_pair() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(0, 2))]
    cfg.exclude_concept_modes = ["7:positive"]
    table = prepare_bpb_attack_rows(
        prefixes=[
            _prefix(concept_id=7, prefix_text="excluded"),
            _prefix(concept_id=1, prefix_text="kept"),
        ],
        work_specs=[_spec()],
        cfg=cfg,
    )
    concept_ids = {row["concept_id"] for row in table.to_pylist()}
    assert 7 not in concept_ids, "excluded pair must emit ZERO rows"
    assert 1 in concept_ids, "other concepts must be untouched"


def test_exclusion_is_mode_scoped_not_concept_wide() -> None:
    """(37, positive) excluded must NOT drop (37, negative)."""
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(0, 2))]
    cfg.exclude_concept_modes = ["7:positive"]
    table = prepare_bpb_attack_rows(
        prefixes=[
            _prefix(concept_id=7, mode=SteeringMode.POSITIVE, prefix_text="pos"),
            _prefix(concept_id=7, mode=SteeringMode.NEGATIVE, prefix_text="neg"),
        ],
        work_specs=[_spec(), _spec(mode=SteeringMode.NEGATIVE)],
        cfg=cfg,
    )
    by_mode = {(row["concept_id"], row["mode"]) for row in table.to_pylist()}
    assert (7, "positive") not in by_mode
    assert (7, "negative") in by_mode, (
        "mode-scoped exclusion must keep the OTHER mode of the same concept"
    )


def test_exclusion_matching_nothing_raises_typo_guard() -> None:
    """'93:positive' with no concept 93 is a typo, not a no-op."""
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(0, 2))]
    cfg.exclude_concept_modes = ["93:positive"]
    with pytest.raises(AxBenchBpbAdvError, match="matched no prefixes"):
        prepare_bpb_attack_rows(
            prefixes=[_prefix(concept_id=1)], work_specs=[_spec()], cfg=cfg
        )


def test_excluding_every_prefix_raises() -> None:
    cfg = _cfg()
    cfg.adversarial_methods = [FasterGcgConfig(optimized_token_counts=(0, 2))]
    cfg.exclude_concept_modes = ["1:positive"]
    with pytest.raises(AxBenchBpbAdvError, match="removed every prefix"):
        prepare_bpb_attack_rows(
            prefixes=[_prefix(concept_id=1)], work_specs=[_spec()], cfg=cfg
        )


def test_malformed_exclusion_entries_rejected_at_config_time() -> None:
    from steering.eval.axbench_bpb_adv.config import validate_config

    for bad in ("37", "positive:37", "37:pos", "x:positive", "37:positive:extra"):
        cfg = _cfg()
        cfg.exclude_concept_modes = [bad]
        with pytest.raises(AxBenchBpbAdvError, match="exclude_concept_modes"):
            validate_config(cfg)


def test_empty_exclusions_do_not_change_run_tag_identity() -> None:
    """Every existing auto-generated tag must survive byte-for-byte."""
    from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict

    payload = axbench_bpb_adv_config_to_dict(_cfg())
    assert "exclude_concept_modes" not in payload


def test_nonempty_exclusions_participate_in_run_tag_identity() -> None:
    """Excluding concepts changes the scored population; tags must differ."""
    from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict

    cfg = _cfg()
    cfg.exclude_concept_modes = ["37:positive"]
    payload = axbench_bpb_adv_config_to_dict(cfg)
    assert payload.get("exclude_concept_modes") == ["37:positive"]
