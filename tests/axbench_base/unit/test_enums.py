"""Unit tests for the review-fix enums.

Pins the :class:`StrEnum` contract that the rest of the pipeline leans
on: bare-string equality and hash must match the enum member, so any
``if mode == "positive"`` checks keep resolving to the same thing whether
the caller passed the enum or the string.

Also checks a handful of drift-risk details:

* The enums subclass :class:`str` (required for parquet / JSON
  round-trip and :class:`pyarrow` interop).
* Every registry modifier key is a :class:`PartitionModifier` member,
  and lookup by the equivalent bare string still works.
* :class:`Stage` — added for the tasks-41-48 review fix — replaces the
  bare stage identifier strings (``"0a"``, ``"3d"``, ``"all"``) used
  by the DAG dep checks and the CLI dispatcher. The test suite pins
  the stable string values, the ``ALL`` pseudo-stage contract, and
  the backward-compat guarantees around dict lookup with string keys.
"""

from __future__ import annotations

from enum import StrEnum
from typing import cast

import pytest

from scripts.eval.axbench_base_eval import _resolve_stage_list
from steering.eval.axbench_base.dag import STAGE_DEPENDENCIES
from steering.eval.axbench_base.encoder_input import (
    _MODIFIER_REGISTRY,
    PartitionModifier,
)
from steering.eval.axbench_base.method_registry import BaselineKind
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Stage,
    SteeringMode,
)

# ---------------------------------------------------------------------------
# Basic class contracts
# ---------------------------------------------------------------------------


def test_steering_mode_is_str_enum() -> None:
    assert issubclass(SteeringMode, StrEnum)
    assert issubclass(SteeringMode, str)


def test_baseline_kind_is_str_enum() -> None:
    assert issubclass(BaselineKind, StrEnum)
    assert issubclass(BaselineKind, str)


def test_partition_modifier_is_str_enum() -> None:
    assert issubclass(PartitionModifier, StrEnum)
    assert issubclass(PartitionModifier, str)


# ---------------------------------------------------------------------------
# String equality / hash — the critical runtime compat contract
# ---------------------------------------------------------------------------


def test_steering_mode_string_equality() -> None:
    assert SteeringMode.POSITIVE == "positive"
    assert SteeringMode.NEGATIVE == "negative"
    assert SteeringMode.POSITIVE != "negative"
    assert SteeringMode.NEGATIVE != "positive"
    # Hash equality — required for set/dict membership with either form.
    assert hash(SteeringMode.POSITIVE) == hash("positive")
    assert hash(SteeringMode.NEGATIVE) == hash("negative")


def test_baseline_kind_string_equality() -> None:
    assert BaselineKind.DIFFMEAN == "diffmean"
    assert BaselineKind.LINEAR_PROBE == "linear_probe"
    assert BaselineKind.LORA == "lora"
    assert BaselineKind.PROMPTING == "prompting"
    assert BaselineKind.OURS == "ours"
    # Hash equality.
    for member in BaselineKind:
        assert hash(member) == hash(str(member)), member


def test_partition_modifier_string_equality() -> None:
    pairs: list[tuple[PartitionModifier, str]] = [
        (PartitionModifier.OURS_EXTRACT_AND_ADD, "ours_extract_and_add"),
        (PartitionModifier.OURS_REPLACE_RANDOM, "ours_replace_random"),
        (PartitionModifier.OURS_REPLACE_ALL, "ours_replace_all"),
        (PartitionModifier.NEG_OURS_REMOVE, "neg_ours_remove"),
        (PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES, "neg_ours_add_to_negatives"),
        (
            PartitionModifier.NEG_OURS_REPLACE_RANDOM_NEGATIVES,
            "neg_ours_replace_random_negatives",
        ),
        (
            PartitionModifier.NEG_OURS_REPLACE_ALL_NEGATIVES,
            "neg_ours_replace_all_negatives",
        ),
        (PartitionModifier.OURS_GCG_ENCDEC, "ours_gcg_encdec"),
        (PartitionModifier.NEG_OURS_GCG_ENCDEC, "neg_ours_gcg_encdec"),
        (PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC, "ours_gcg_roofline_encdec"),
        (
            PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC,
            "neg_ours_gcg_roofline_encdec",
        ),
        (PartitionModifier.FEATURE_DIFFMEAN_ENCDEC, "feature_diffmean_encdec"),
        (
            PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC,
            "neg_feature_diffmean_encdec",
        ),
        (
            PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
            "feature_diffmean_roofline_encdec",
        ),
        (
            PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
            "neg_feature_diffmean_roofline_encdec",
        ),
    ]
    for member, expected_str in pairs:
        assert member == expected_str, member
        assert hash(member) == hash(expected_str), member


# ---------------------------------------------------------------------------
# Dict lookup via the enum or the equivalent string must agree
# ---------------------------------------------------------------------------


def test_dict_lookup_with_enum_key_accepts_string() -> None:
    """StrEnum guarantees the hash/equality match, so dict lookup with
    either key form resolves to the same entry. We pass both forms
    through ``cast`` so ``ty`` accepts the str-keyed call; the runtime
    semantics are unchanged because :class:`PartitionModifier` members
    hash equal to their string values.
    """
    for member in PartitionModifier:
        via_enum = _MODIFIER_REGISTRY.get(member)
        via_str = _MODIFIER_REGISTRY.get(cast(PartitionModifier, str(member)))
        assert via_enum is not None, member
        assert via_enum is via_str, member


def test_every_registry_key_is_a_partition_modifier_member() -> None:
    for key in _MODIFIER_REGISTRY:
        assert isinstance(key, PartitionModifier), key


# ---------------------------------------------------------------------------
# Stage — basic class + member contracts
# ---------------------------------------------------------------------------


def test_stage_is_str_enum() -> None:
    assert issubclass(Stage, StrEnum)
    assert issubclass(Stage, str)


def test_stage_members_exist() -> None:
    """Guard against accidental rename / deletion of Stage members.

    This test doubles as a backward-compatibility contract: the member
    VALUES are baked into Hydra YAML configs (``stage: all``), CLI
    strings passed by operators, and historical logs / shell scripts.
    Changing a value is a breaking change; changing a NAME is safe.
    """
    expected_values = {
        "PREPARE_CONCEPTS": "0a",
        "PREPARE_SEED_POOLS": "0b",
        "PREPARE_NEGATIVE_PREFIXES": "0c",
        "SYNTH_POSITIVE_MODE": "1b",
        "SYNTH_NEGATIVE_MODE": "1c",
        "CONCEPT_FEATURE_MAPPING": "2",
        "TRAIN_DIFFMEAN": "3a",
        "TRAIN_LINEAR_PROBE": "3b",
        "TRAIN_PROMPTING": "3c",
        "TRAIN_LORA": "3d",
        "ENUMERATE_STAGE4_CONTEXTS": "3e",
        "GENERATE": "4",
        "JUDGE": "5",
        "AGGREGATE": "6",
        "ALL": "all",
    }
    assert {m.name: m.value for m in Stage} == expected_values


def test_stage_values_are_stable_identifiers() -> None:
    """The 15 short string IDs the operator pipeline depends on are
    present and stable. This is a tighter contract than the name→value
    map above: it says specific strings must exist as Stage values.

    Stage ``1a`` (SYNTH_GLOBAL_NEGATIVES) was removed alongside the
    global-negative pool — a deliberate breaking change to the CLI id
    set."""
    values = {s.value for s in Stage}
    assert values == {
        "0a",
        "0b",
        "0c",
        "1b",
        "1c",
        "2",
        "3a",
        "3b",
        "3c",
        "3d",
        "3e",
        "4",
        "5",
        "6",
        "all",
    }


def test_stage_enum_has_14_functional_members_plus_all() -> None:
    """The Stage enum has 14 functional stages + ``ALL`` pseudo-stage.

    Stage 3e adds exact Stage 4 context enumeration after LoRA training.
    Pins the exact count so accidental additions / deletions
    fail loudly in CI rather than quietly shifting the pipeline shape.
    """
    all_members = list(Stage)
    assert len(all_members) == 15
    functional = [m for m in all_members if m is not Stage.ALL]
    assert len(functional) == 14


def test_stage_train_diffmean_has_value_3a() -> None:
    assert Stage.TRAIN_DIFFMEAN.value == "3a"


def test_stage_train_linear_probe_has_value_3b() -> None:
    assert Stage.TRAIN_LINEAR_PROBE.value == "3b"


def test_stage_train_prompting_has_value_3c() -> None:
    assert Stage.TRAIN_PROMPTING.value == "3c"


def test_stage_train_lora_has_value_3d() -> None:
    """Regression: the old pre-split enum had ``TRAIN_LORA = "3b"``,
    which now belongs to :attr:`Stage.TRAIN_LINEAR_PROBE`. LoRA moved
    to ``"3d"`` after the Stage 3 split.
    """
    assert Stage.TRAIN_LORA.value == "3d"


def test_stage_coerce_3b_is_linear_probe_not_lora() -> None:
    """Regression guard: before the Stage 3 split the value ``"3b"``
    resolved to ``Stage.TRAIN_LORA``. After the split, ``"3b"`` is
    ``Stage.TRAIN_LINEAR_PROBE`` and LoRA lives at ``"3d"``. A
    resurrection of the old mapping would silently re-introduce the
    fall-through bug that motivated the split in the first place.
    """
    assert Stage.coerce("3b") is Stage.TRAIN_LINEAR_PROBE
    assert Stage.coerce("3b") is not Stage.TRAIN_LORA
    assert Stage.coerce("3d") is Stage.TRAIN_LORA


def test_stage_coerce_accepts_int_for_numeric_stages() -> None:
    """Operators on the Hydra CLI write ``stage=2`` to target Stage
    ``CONCEPT_FEATURE_MAPPING``. OmegaConf type-infers bare-numeric
    values to :class:`int`, so ``Stage.coerce(2)`` must resolve to
    the same enum member as ``Stage.coerce("2")`` — otherwise the
    operator sees "Unknown stage 2" while the known-stages list
    visibly contains ``'2'``. The regression killed the
    ``dev-3targets-d12_baseline-20260421-221023`` chain mid-run; see
    PR history.

    Equally critical: the StrEnum's hash parity contract ensures
    ``Stage.coerce(2) == "2"`` stays true. A future refactor of
    ``_run_single_stage`` that compares ``stage_enum == 2`` (rather
    than the current string comparison) would silently break without
    this test.
    """
    assert Stage.coerce(2) is Stage.CONCEPT_FEATURE_MAPPING
    assert Stage.coerce("2") is Stage.CONCEPT_FEATURE_MAPPING
    assert Stage.coerce(4) is Stage.GENERATE
    assert Stage.coerce(5) is Stage.JUDGE
    assert Stage.coerce(6) is Stage.AGGREGATE


def test_stage_coerce_does_not_accept_bool() -> None:
    """In Python, ``bool`` is a subclass of ``int`` — ``isinstance(True, int)``
    returns ``True``. The int-normalisation branch in
    :meth:`Stage.coerce` explicitly excludes ``bool`` so
    ``Stage.coerce(True)`` does not silently resolve to some stage
    whose value happens to be ``"1"``.
    """
    with pytest.raises(AxBenchBaseError, match="Unknown stage"):
        Stage.coerce(True)
    with pytest.raises(AxBenchBaseError, match="Unknown stage"):
        Stage.coerce(False)


def test_stage_coerce_unknown_int_raises_with_listing() -> None:
    """Non-matching ints surface the same helpful error as non-matching
    strings: the original value is shown (``repr``'d) plus the full
    list of known identifiers."""
    with pytest.raises(AxBenchBaseError) as exc_info:
        Stage.coerce(99)
    message = str(exc_info.value)
    assert "99" in message
    assert "'0a'" in message
    assert "'all'" in message


# ---------------------------------------------------------------------------
# Stage — string equality / hash parity (the critical runtime compat contract)
# ---------------------------------------------------------------------------


def test_stage_string_equality() -> None:
    assert Stage.PREPARE_CONCEPTS == "0a"
    assert Stage.PREPARE_SEED_POOLS == "0b"
    assert Stage.PREPARE_NEGATIVE_PREFIXES == "0c"
    assert Stage.SYNTH_POSITIVE_MODE == "1b"
    assert Stage.SYNTH_NEGATIVE_MODE == "1c"
    assert Stage.CONCEPT_FEATURE_MAPPING == "2"
    assert Stage.TRAIN_DIFFMEAN == "3a"
    assert Stage.TRAIN_LINEAR_PROBE == "3b"
    assert Stage.TRAIN_PROMPTING == "3c"
    assert Stage.TRAIN_LORA == "3d"
    assert Stage.ENUMERATE_STAGE4_CONTEXTS == "3e"
    assert Stage.GENERATE == "4"
    assert Stage.JUDGE == "5"
    assert Stage.AGGREGATE == "6"
    assert Stage.ALL == "all"
    # Non-equality: different members are not equal.
    assert Stage.PREPARE_CONCEPTS != "0b"
    assert Stage.TRAIN_DIFFMEAN != "3b"
    assert Stage.TRAIN_LORA != "3b"


def test_stage_hash_equality() -> None:
    """StrEnum guarantees ``hash(Stage.X) == hash(Stage.X.value)``, which
    is what makes dict lookup with a string key work for enum-keyed dicts.
    """
    for member in Stage:
        assert hash(member) == hash(member.value), member


# ---------------------------------------------------------------------------
# Stage — dict lookup via string key still resolves to the enum entry
# ---------------------------------------------------------------------------


def test_stage_dict_lookup_accepts_string() -> None:
    """``STAGE_DEPENDENCIES.get("0a")`` returns the same value as
    ``STAGE_DEPENDENCIES.get(Stage.PREPARE_CONCEPTS)``.

    This is the core backward-compat guarantee: every call site that
    currently passes a bare string into ``STAGE_DEPENDENCIES`` keeps
    working, because StrEnum members hash-equal their string values.
    """
    for member in Stage:
        if member is Stage.ALL:
            # ALL is the pseudo-stage and is intentionally absent from
            # the dep dict; validated separately below.
            continue
        via_enum = STAGE_DEPENDENCIES.get(member)
        via_str = STAGE_DEPENDENCIES.get(cast(Stage, member.value))
        assert via_enum is not None, member
        assert via_enum is via_str, member


def test_stage_all_is_pseudo() -> None:
    """``Stage.ALL`` is the "run every stage" sentinel and must NOT be
    a key in ``STAGE_DEPENDENCIES``. The CLI dispatcher expands it to
    the real stage list before the dep check sees it.
    """
    assert Stage.ALL not in STAGE_DEPENDENCIES
    assert "all" not in STAGE_DEPENDENCIES


def test_every_stage_dependencies_key_is_a_stage_member() -> None:
    """The DAG dict is keyed on :class:`Stage`; every key is an enum
    member and every (real) non-ALL enum member appears as a key."""
    for key in STAGE_DEPENDENCIES:
        assert isinstance(key, Stage), key
    real_members = {m for m in Stage if m is not Stage.ALL}
    assert set(STAGE_DEPENDENCIES.keys()) == real_members


# ---------------------------------------------------------------------------
# Stage — unknown / malformed inputs
# ---------------------------------------------------------------------------


def test_unknown_stage_value_raises_at_construction() -> None:
    """``Stage("bogus")`` must raise a plain :class:`ValueError` — the
    CLI wraps it in :class:`AxBenchBaseError` before surfacing to
    operators, but at the enum level we want the stdlib behavior.
    """
    with pytest.raises(ValueError):
        Stage("bogus")


def test_resolve_stage_list_accepts_stage_member_for_all() -> None:
    """``_resolve_stage_list`` should accept either a raw string or a
    :class:`Stage` member. Passing ``Stage.ALL`` expands the same way
    as passing the literal ``"all"`` string.
    """
    out = _resolve_stage_list(Stage.ALL, run_negative_mode=True)
    assert Stage.SYNTH_NEGATIVE_MODE in out
    out_via_str = _resolve_stage_list("all", run_negative_mode=True)
    assert out == out_via_str


def test_resolve_stage_list_skips_1c_when_neg_mode_off() -> None:
    """The ``run_negative_mode=False`` skip rule removes exactly the
    ``SYNTH_NEGATIVE_MODE`` stage from the expanded ``Stage.ALL``
    list. Every other stage stays in order.
    """
    out = _resolve_stage_list(Stage.ALL, run_negative_mode=False)
    assert Stage.SYNTH_NEGATIVE_MODE not in out
    assert len(out) == len([m for m in Stage if m is not Stage.ALL]) - 1
    # Order is preserved modulo the 1c deletion.
    real_order = [m for m in Stage if m is not Stage.ALL]
    expected = [m for m in real_order if m is not Stage.SYNTH_NEGATIVE_MODE]
    assert out == expected


def test_resolve_stage_list_unknown_raises_axbench_error() -> None:
    """At the CLI surface, an unknown stage name raises
    :class:`AxBenchBaseError` (not a raw ``ValueError``)."""
    with pytest.raises(AxBenchBaseError, match="Unknown stage"):
        _resolve_stage_list("7z", run_negative_mode=True)
