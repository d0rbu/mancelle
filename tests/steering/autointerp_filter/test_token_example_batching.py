from __future__ import annotations

import hashlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.eval.autointerp_filter.token_examples as token_examples_module
from steering.eval.autointerp_filter.batch_validation import (
    BatchCalibrationContract,
    BatchSurfaceKind,
    BatchToleranceProfile,
    NumericTolerance,
)
from steering.eval.autointerp_filter.batching import (
    MaterializationPhase,
    PreparedBatch,
    PreparedDoc,
    TargetBlock,
    build_target_blocks,
    target_plan_hash,
)
from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    _DocActivations,
    _exception_chain_contains_cuda_oom,
    _make_token_window_ref,
    _non_activating_windows_from_doc,
    _positive_windows_from_doc,
    _project_selected_sae_activations,
    _selected_sae_activations,
    _target_activation_column_from_compact,
    _TargetBuildState,
    _validate_selected_sae_inputs,
)
from steering.eval.autointerp_filter.types import (
    AutoInterpBatchValidationError,
    AutoInterpDataError,
    ScoringTarget,
    SourceDocRef,
    TargetKind,
)
from steering.refined_types import PositiveInt


def _target(feature_id: int, *, suffix: str = "") -> ScoringTarget:
    key = f"feature:{feature_id}{suffix}"
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(feature_id,),
        target_key=key,
        display_name=key.replace(":", "_"),
    )


def _doc_ref(index: int, *, start: int = 0, end: int = 8) -> SourceDocRef:
    return SourceDocRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=index,
        segment_token_start=start,
        segment_token_end=end,
    )


def _prepared_doc(
    *,
    block_ordinal: int = 0,
    slot: int = 1,
    feature_id: int = 1,
    target_key: str = "feature:1",
    source_text_hash: str | None = None,
    model_token_ids: tuple[int, ...] = (2, 11, 12, 13),
    real_token_ids: tuple[int, ...] = (11, 12, 13),
    token_strings: tuple[str, ...] = ("a", "b", "c"),
    activation_slice_start: int = 1,
    token_offset: int = 4,
) -> PreparedDoc:
    return PreparedDoc(
        block_ordinal=block_ordinal,
        slot=slot,
        target_key=target_key,
        feature_id=feature_id,
        phase=MaterializationPhase.positive,
        doc_ref=_doc_ref(0, end=3),
        source_text_hash=(f"hash-{feature_id}" if source_text_hash is None else source_text_hash),
        model_token_ids=model_token_ids,
        real_token_ids=real_token_ids,
        token_strings=token_strings,
        token_offset=token_offset,
        activation_slice_start=activation_slice_start,
    )


def _doc_activations(index: int, values: tuple[float, ...]) -> _DocActivations:
    return _DocActivations(
        source_text_hash=f"hash-{index}",
        token_offset=index * 10,
        token_ids=tuple(1000 + index * 100 + offset for offset in range(len(values))),
        token_strings=tuple(f"tok-{index}-{offset}" for offset in range(len(values))),
        sae_feature_ids=(7,),
        sae_acts=t.tensor(values, dtype=t.float32).unsqueeze(1),
    )


@settings(max_examples=100, deadline=None)
@given(
    feature_ids=st.sets(st.integers(min_value=0, max_value=2047), min_size=1, max_size=80),
    batch_size=st.integers(min_value=1, max_value=64),
    shuffle_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_target_block_mapping_is_total_unique_and_feature_addressed(
    feature_ids: set[int],
    batch_size: int,
    shuffle_seed: int,
) -> None:
    ids = list(feature_ids)
    random.Random(shuffle_seed).shuffle(ids)
    targets = [_target(feature_id) for feature_id in ids]

    blocks = build_target_blocks(targets, batch_size=batch_size)

    assert [block.block_id for block in blocks] == sorted({fid // batch_size for fid in ids})
    assert [block.ordinal for block in blocks] == list(range(len(blocks)))
    recovered: dict[int, ScoringTarget] = {}
    for block in blocks:
        assert len(block.targets_by_slot) == batch_size
        for slot, target in enumerate(block.targets_by_slot):
            if target is None:
                continue
            feature_id = target.feature_ids[0]
            assert divmod(feature_id, batch_size) == (block.block_id, slot)
            assert feature_id not in recovered
            recovered[feature_id] = target
    assert recovered == {target.feature_ids[0]: target for target in targets}


def test_target_plan_hash_pins_order_and_every_target_field() -> None:
    targets = [_target(1), _target(129)]
    baseline = target_plan_hash(targets)
    assert target_plan_hash(list(targets)) == baseline
    assert target_plan_hash(list(reversed(targets))) != baseline
    assert target_plan_hash([replace(targets[0], display_name="different"), targets[1]]) != baseline
    assert target_plan_hash([replace(targets[0], target_key="different"), targets[1]]) != baseline


@pytest.mark.parametrize("bad_batch_size", [True, False, 0, -1, 1.5, "2", None])
def test_build_target_blocks_rejects_non_strict_or_nonpositive_batch_size(
    bad_batch_size: Any,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="batch_size"):
        build_target_blocks([_target(1)], batch_size=bad_batch_size)


def test_build_target_blocks_rejects_wrong_target_collection_items() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="targets"):
        build_target_blocks(cast(Any, [_target(0), object()]), batch_size=2)


def test_build_target_blocks_rejects_duplicate_key_feature_and_nonsingleton() -> None:
    with pytest.raises(AutoInterpDataError, match="duplicate target keys"):
        build_target_blocks(
            [_target(1), replace(_target(2), target_key="feature:1")],
            batch_size=2,
        )

    same_feature = replace(_target(1), target_key="second-key")
    with pytest.raises(AutoInterpDataError, match="same deterministic.*slot"):
        build_target_blocks([_target(1), same_feature], batch_size=2)

    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        replace(_target(1), feature_ids=(1, 2))

    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        replace(_target(1), feature_ids=(-1,))


def test_target_block_validates_shape_identity_and_nonempty_slots() -> None:
    with pytest.raises(AutoInterpDataError, match="slot count"):
        TargetBlock(ordinal=0, block_id=0, batch_size=2, targets_by_slot=(_target(0),))
    with pytest.raises(AutoInterpDataError, match="entirely empty"):
        TargetBlock(ordinal=0, block_id=0, batch_size=2, targets_by_slot=(None, None))
    with pytest.raises(AutoInterpDataError, match="wrong deterministic.*slot"):
        TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=2,
            targets_by_slot=(None, _target(0)),
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="ordinal"):
        TargetBlock(ordinal=-1, block_id=0, batch_size=1, targets_by_slot=(_target(0),))
    with pytest.raises(BeartypeCallHintParamViolation, match="block_id"):
        TargetBlock(ordinal=0, block_id=-1, batch_size=1, targets_by_slot=(_target(0),))
    with pytest.raises(BeartypeCallHintParamViolation, match="batch_size"):
        TargetBlock(ordinal=0, block_id=0, batch_size=0, targets_by_slot=())
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        replace(_target(0), feature_ids=(0, 1))


@pytest.mark.parametrize(
    ("field_name", "bad"),
    [
        ("ordinal", True),
        ("ordinal", 0.0),
        ("block_id", False),
        ("block_id", "0"),
        ("batch_size", True),
        ("batch_size", 1.0),
    ],
)
def test_target_block_rejects_non_strict_integer_identity_fields(
    field_name: str,
    bad: Any,
) -> None:
    values: dict[str, Any] = {
        "ordinal": 0,
        "block_id": 0,
        "batch_size": 1,
        "targets_by_slot": (_target(0),),
    }
    values[field_name] = bad
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        TargetBlock(**values)


def test_target_block_rejects_mutable_invalid_and_duplicate_target_slots() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="targets_by_slot"):
        TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=1,
            targets_by_slot=cast(Any, [_target(0)]),
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="targets_by_slot"):
        TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=1,
            targets_by_slot=cast(Any, (object(),)),
        )
    duplicate_key = replace(_target(1), target_key="feature:0")
    with pytest.raises(AutoInterpDataError, match="unique"):
        TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=2,
            targets_by_slot=(_target(0), duplicate_key),
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        replace(_target(0), feature_ids=(True,))


def test_prepared_doc_accepts_natural_bos_and_explicit_injected_bos() -> None:
    injected = _prepared_doc()
    natural = _prepared_doc(
        model_token_ids=(2, 11, 12),
        real_token_ids=(2, 11, 12),
        token_strings=("<bos>", "a", "b"),
        activation_slice_start=0,
    )
    assert injected.model_token_ids[1:] == injected.real_token_ids
    assert natural.model_token_ids == natural.real_token_ids


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"block_ordinal": -1}, "ordinal and slot"),
        ({"slot": -1}, "ordinal and slot"),
        ({"feature_id": -1}, "feature id"),
        ({"target_key": ""}, "target key"),
        ({"source_text_hash": ""}, "source hash"),
        ({"token_offset": -1}, "token offset"),
        ({"activation_slice_start": 2}, "exactly 0 or 1"),
        ({"real_token_ids": (), "token_strings": (), "model_token_ids": (2,)}, "real token"),
        ({"token_strings": ("too", "short")}, "length mismatch"),
        ({"model_token_ids": (2, 11, 12)}, "surface length"),
        ({"model_token_ids": (2, 99, 12, 13)}, "does not contain"),
    ],
)
def test_prepared_doc_rejects_every_inconsistent_surface(
    updates: dict[str, Any],
    message: str,
) -> None:
    values = {
        "block_ordinal": 0,
        "slot": 1,
        "feature_id": 1,
        "target_key": "feature:1",
        "model_token_ids": (2, 11, 12, 13),
        "real_token_ids": (11, 12, 13),
        "token_strings": ("a", "b", "c"),
        "activation_slice_start": 1,
    }
    values.update(updates)
    field_name = next(iter(updates))
    direct_domain_error = field_name in {
        "block_ordinal",
        "slot",
        "feature_id",
        "target_key",
        "source_text_hash",
        "token_offset",
        "activation_slice_start",
        "real_token_ids",
    }
    error_type = BeartypeCallHintParamViolation if direct_domain_error else AutoInterpDataError
    match = field_name if direct_domain_error else message
    with pytest.raises(error_type, match=match):
        _prepared_doc(**cast(Any, values))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"block_ordinal": True}, "strict non-negative ints"),
        ({"slot": 1.0}, "strict non-negative ints"),
        ({"feature_id": False}, "strict non-negative int"),
        ({"phase": "positive"}, "MaterializationPhase"),
        ({"doc_ref": object()}, "SourceDocRef"),
        ({"token_offset": True}, "strict non-negative int"),
        ({"activation_slice_start": False}, "exactly 0 or 1"),
        ({"model_token_ids": [2, 11, 12, 13]}, "immutable tuples"),
        ({"real_token_ids": [11, 12, 13]}, "immutable tuples"),
        ({"token_strings": ["a", "b", "c"]}, "immutable tuples"),
        ({"model_token_ids": (2, -1, 12, 13)}, "strict non-negative ints"),
        ({"real_token_ids": (True, 12, 13)}, "strict non-negative ints"),
        ({"token_strings": ("a", 2, "c")}, "must all be strings"),
    ],
)
def test_prepared_doc_rejects_mutable_or_non_strict_runtime_types(
    updates: dict[str, Any],
    message: str,
) -> None:
    del message
    with pytest.raises(BeartypeCallHintParamViolation, match=next(iter(updates))):
        replace(_prepared_doc(), **updates)


def test_prepared_batch_tensor_is_fixed_shape_right_padded_and_contiguous() -> None:
    block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=4,
        targets_by_slot=(_target(0), _target(1), None, _target(3)),
    )
    rows = (
        _prepared_doc(
            slot=0,
            feature_id=0,
            target_key="feature:0",
            model_token_ids=(2, 10),
            real_token_ids=(10,),
            token_strings=("x",),
        ),
        _prepared_doc(slot=1, feature_id=1, target_key="feature:1"),
        None,
        _prepared_doc(
            slot=3,
            feature_id=3,
            target_key="feature:3",
            model_token_ids=(20, 21),
            real_token_ids=(20, 21),
            token_strings=("y", "z"),
            activation_slice_start=0,
        ),
    )
    prepared = PreparedBatch(block=block, rows=rows, model_width=5)

    tokens = prepared.token_tensor()
    second = prepared.token_tensor()
    assert second.data_ptr() != tokens.data_ptr()
    assert t.equal(second, tokens)

    assert prepared.has_work
    assert tokens.dtype == t.long
    assert tokens.device.type == "cpu"
    assert tokens.is_contiguous()
    assert tokens.tolist() == [
        [2, 10, 0, 0, 0],
        [2, 11, 12, 13, 0],
        [0, 0, 0, 0, 0],
        [20, 21, 0, 0, 0],
    ]
    tokens.zero_()
    assert prepared.token_tensor().tolist()[0] == [2, 10, 0, 0, 0]


def test_prepared_batch_rejects_row_slot_target_width_and_pad_mismatches() -> None:
    block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=2,
        targets_by_slot=(_target(0), _target(1)),
    )
    row0 = _prepared_doc(slot=0, feature_id=0, target_key="feature:0")
    row1 = _prepared_doc(slot=1, feature_id=1, target_key="feature:1")
    with pytest.raises(AutoInterpDataError, match="row count"):
        PreparedBatch(block=block, rows=(row0,), model_width=4)
    with pytest.raises(BeartypeCallHintParamViolation, match="model_width"):
        PreparedBatch(block=block, rows=(row0, row1), model_width=0)
    with pytest.raises(BeartypeCallHintParamViolation, match="pad_token_id"):
        PreparedBatch(block=block, rows=(row0, row1), model_width=4, pad_token_id=1)
    with pytest.raises(AutoInterpDataError, match="wrong slot/block"):
        PreparedBatch(block=block, rows=(replace(row0, slot=1), row1), model_width=4)
    with pytest.raises(AutoInterpDataError, match="row target mismatch"):
        PreparedBatch(
            block=block,
            rows=(replace(row0, target_key="wrong"), row1),
            model_width=4,
        )
    with pytest.raises(AutoInterpDataError, match="exceeds fixed"):
        PreparedBatch(block=block, rows=(row0, row1), model_width=3)
    with pytest.raises(BeartypeCallHintParamViolation, match="rows"):
        PreparedBatch(
            block=block,
            rows=cast(Any, [row0, row1]),
            model_width=4,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="rows"):
        PreparedBatch(
            block=block,
            rows=cast(Any, (row0, object())),
            model_width=4,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="model_width"):
        PreparedBatch(block=block, rows=(row0, row1), model_width=True)
    with pytest.raises(BeartypeCallHintParamViolation, match="pad_token_id"):
        PreparedBatch(block=block, rows=(row0, row1), model_width=4, pad_token_id=False)

    sparse_block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=2,
        targets_by_slot=(_target(0), None),
    )
    with pytest.raises(AutoInterpDataError, match="work but no block target"):
        PreparedBatch(block=sparse_block, rows=(row0, row1), model_width=4)


def test_pure_window_builders_pin_order_bounds_activity_and_caps() -> None:
    target = _target(7)
    doc_ref = _doc_ref(3, start=4, end=12)
    doc = _doc_activations(3, (0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0))

    positives = _positive_windows_from_doc(
        doc_ref,
        doc,
        target,
        example_ctx_len=4,
        max_windows_per_source_doc=2,
    )
    non_activating = _non_activating_windows_from_doc(
        doc_ref,
        doc,
        target,
        example_ctx_len=2,
        max_windows_per_source_doc=8,
    )

    assert len(positives) == 2
    assert [ref.max_activation for ref in positives] == [5.0, 5.0]
    assert all(ref.active for ref in positives)
    assert [(ref.token_start, ref.token_end) for ref in positives] == [(30, 34), (33, 37)]
    assert [(ref.token_start, ref.token_end) for ref in non_activating] == [(34, 36)]
    assert all(not ref.active and ref.max_activation == 0.0 for ref in non_activating)
    capped_zero = _non_activating_windows_from_doc(
        doc_ref,
        _doc_activations(3, (0.0,) * 8),
        target,
        example_ctx_len=2,
        max_windows_per_source_doc=2,
    )
    assert len(capped_zero) == 2


def test_pure_window_builders_reject_invalid_target_and_missing_compact_columns() -> None:
    doc_ref = _doc_ref(3)
    doc = _doc_activations(3, (0.0, 1.0, 0.0, 0.0))
    with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
        replace(_target(7), feature_ids=(7, 8))
    with pytest.raises(AutoInterpDataError, match="missing from compact SAE"):
        _non_activating_windows_from_doc(
            doc_ref,
            replace(doc, sae_feature_ids=(8,)),
            _target(7),
            example_ctx_len=4,
            max_windows_per_source_doc=1,
        )


def test_compact_column_and_ref_helpers_share_the_serial_policy() -> None:
    target = _target(7)
    doc_ref = _doc_ref(3, start=4, end=8)
    doc = _doc_activations(3, (0.0, 2.0, 0.0, 1.0))

    column = _target_activation_column_from_compact(
        doc.sae_acts,
        doc.sae_feature_ids,
        target,
    )
    ref = _make_token_window_ref(
        doc_ref,
        doc,
        target,
        start=1,
        end=3,
        activations=column[1:3],
        active=True,
    )

    assert column.tolist() == [0.0, 2.0, 0.0, 1.0]
    assert ref.token_ids == doc.token_ids[1:3]
    assert ref.token_strings == doc.token_strings[1:3]
    assert ref.token_start == doc.token_offset + 1
    assert ref.token_end == doc.token_offset + 3
    assert ref.activations == (2.0, 0.0)
    assert ref.max_activation == 2.0
    assert ref.target_feature_ids == (7,)
    assert ref.active is True


def _state_fixture(
    *,
    tmp_path: Path,
    seed: int,
    positive_cap: int,
    intruder_cap: int,
) -> tuple[CachedTokenExampleProvider, _TargetBuildState, dict[str, _DocActivations]]:
    target = _target(7)
    positive_refs = [_doc_ref(index) for index in range(5)]
    intruder_refs = [_doc_ref(index) for index in range(3, 12)]
    values = {
        ref.stable_key(): _doc_activations(
            ref.doc_id,
            tuple(
                0.0 if (position + ref.doc_id) % 4 else float((position + ref.doc_id) % 7 + 1)
                for position in range(8)
            ),
        )
        for ref in {*positive_refs, *intruder_refs}
    }
    cfg = AutoInterpFilterConfig(
        feature_dir="fixture/features",
        output_dir=str(tmp_path / "out"),
        seed=seed,
        token_examples=TokenExamplesConfig(
            example_ctx_len=PositiveInt.parse(4),
            max_windows_per_source_doc=PositiveInt.parse(3),
            max_positive_windows_per_target=PositiveInt.parse(positive_cap),
            max_non_activating_windows_per_target=PositiveInt.parse(intruder_cap),
            materialization_batch_size=PositiveInt.parse(1),
        ),
    )
    provider = object.__new__(CachedTokenExampleProvider)
    provider._cfg = cfg
    provider._positive_docs = {target.target_key: list(reversed(positive_refs))}
    provider._positive_doc_scores = {
        target.target_key: {ref.stable_key(): float(ref.doc_id % 3) for ref in positive_refs}
    }
    provider._intruder_docs = list(intruder_refs)
    provider._doc_activations = lambda ref, _feature_ids: values[ref.stable_key()]  # ty: ignore[invalid-assignment]
    state = provider._new_target_build_state(target)
    return provider, state, values


@pytest.mark.parametrize("seed", [0, 1, 17, 2**31 - 1])
@pytest.mark.parametrize("positive_cap", [1, 2, 5, 20])
@pytest.mark.parametrize("intruder_cap", [1, 3, 9])
def test_document_state_machine_is_exactly_equal_to_serial_builder(
    tmp_path: Path,
    seed: int,
    positive_cap: int,
    intruder_cap: int,
) -> None:
    provider, state, values = _state_fixture(
        tmp_path=tmp_path,
        seed=seed,
        positive_cap=positive_cap,
        intruder_cap=intruder_cap,
    )
    serial = provider._build_examples_for_target(state.target)

    while (work := state.next_doc()) is not None:
        phase, doc_ref = work
        state.accept_doc(
            phase=phase,
            doc_ref=doc_ref,
            doc=values[doc_ref.stable_key()],
        )

    assert state.result() == serial


def test_document_state_machine_rejects_double_request_wrong_doc_and_early_result(
    tmp_path: Path,
) -> None:
    _provider, state, values = _state_fixture(
        tmp_path=tmp_path,
        seed=3,
        positive_cap=4,
        intruder_cap=4,
    )
    work = state.next_doc()
    assert work is not None
    with pytest.raises(AutoInterpDataError, match="before accepting"):
        state.next_doc()
    with pytest.raises(AutoInterpDataError, match="unexpected document"):
        state.accept_doc(
            phase=work[0],
            doc_ref=_doc_ref(999),
            doc=_doc_activations(999, (0.0,) * 8),
        )
    with pytest.raises(AutoInterpDataError, match="before completion"):
        state.result()
    state.accept_doc(
        phase=work[0],
        doc_ref=work[1],
        doc=values[work[1].stable_key()],
    )


def test_document_state_machine_rejects_wrong_compact_feature_column(tmp_path: Path) -> None:
    _provider, state, values = _state_fixture(
        tmp_path=tmp_path,
        seed=3,
        positive_cap=4,
        intruder_cap=4,
    )
    work = state.next_doc()
    assert work is not None
    doc = values[work[1].stable_key()]
    with pytest.raises(AutoInterpDataError, match="compact SAE columns"):
        state.accept_doc(
            phase=work[0],
            doc_ref=work[1],
            doc=replace(doc, sae_feature_ids=(8,)),
        )


def test_document_state_machine_dedupes_positive_window_from_intruder_phase() -> None:
    target = _target(7)
    doc_ref = _doc_ref(1, start=0, end=8)
    state = _TargetBuildState(
        target=target,
        positive_docs=(doc_ref,),
        intruder_docs=(doc_ref,),
        example_ctx_len=8,
        max_windows_per_source_doc=2,
        max_positive_windows=2,
        max_non_activating_windows=2,
    )
    positive_work = state.next_doc()
    assert positive_work == (MaterializationPhase.positive, doc_ref)
    state.accept_doc(
        phase=MaterializationPhase.positive,
        doc_ref=doc_ref,
        doc=_doc_activations(1, (0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    )
    intruder_work = state.next_doc()
    assert intruder_work == (MaterializationPhase.intruder, doc_ref)
    state.accept_doc(
        phase=MaterializationPhase.intruder,
        doc_ref=doc_ref,
        doc=_doc_activations(1, (0.0,) * 8),
    )

    assert state.next_doc() is None
    assert state.next_doc() is None
    result = state.result()
    assert len(result.positives) == 1
    assert result.non_activating == ()


class _FakeSAE(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(
            t.tensor(
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 1.0, -1.0],
                ],
                dtype=t.float32,
            )
        )
        self.b_enc = t.nn.Parameter(t.tensor([0.5, -0.5, 1.0], dtype=t.float32))
        self.threshold = t.nn.Parameter(
            t.tensor([0.0, 0.25, 1.5], dtype=t.float32),
            requires_grad=False,
        )
        self.activation_fn = t.nn.ReLU()

    @staticmethod
    def process_sae_in(acts: t.Tensor) -> t.Tensor:
        return acts


def test_compact_projection_kernel_matches_selected_public_wrapper_exactly() -> None:
    sae = _FakeSAE()
    acts = t.tensor([[1.0, 2.0], [-1.0, 3.0], [4.0, -2.0]], dtype=t.float32)
    target_ids = t.tensor([2, 0], dtype=t.long)

    direct = _project_selected_sae_activations(
        sae=sae,
        acts=acts,
        target_ids=target_ids,
    )
    wrapped = _selected_sae_activations(sae=sae, acts=acts, feature_ids=(2, 0))

    assert t.equal(direct, wrapped)
    assert direct.tolist() == [[0.0, 1.5], [0.0, 0.0], [11.0, 4.5]]


def test_projection_validation_rejects_shape_empty_negative_bool_and_upper_bound() -> None:
    sae = _FakeSAE()
    acts = t.zeros((2, 2), dtype=t.float32)
    for feature_ids in ((), (-1,), (True,)):
        with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
            _validate_selected_sae_inputs(sae=sae, acts=acts, feature_ids=feature_ids)
    with pytest.raises(AutoInterpDataError, match="expected 2D"):
        _validate_selected_sae_inputs(sae=sae, acts=acts.unsqueeze(0), feature_ids=(0,))
    with pytest.raises(AutoInterpDataError, match="out of range"):
        _validate_selected_sae_inputs(sae=sae, acts=acts, feature_ids=(3,))


def test_projection_validation_does_not_call_tensor_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sae = _FakeSAE()
    acts = t.zeros((2, 2), dtype=t.float32)

    def _forbid_item(_self: t.Tensor) -> float:
        raise AssertionError("validation introduced a host-synchronizing Tensor.item()")

    monkeypatch.setattr(t.Tensor, "item", _forbid_item)
    _validate_selected_sae_inputs(sae=sae, acts=acts, feature_ids=(0, 2))


class _NativeModel:
    device = t.device("cpu")


class _ExecutorSAE(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(
            t.tensor(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=t.float32,
            )
        )
        self.b_enc = t.nn.Parameter(t.zeros(3, dtype=t.float32))
        self.activation_fn = t.nn.ReLU()

    @staticmethod
    def process_sae_in(acts: t.Tensor) -> t.Tensor:
        return acts

    @staticmethod
    def encode(_acts: t.Tensor) -> t.Tensor:
        raise AssertionError("batch executor must never full-encode the SAE")


def _executor_provider(*, max_abs: float = 0.0) -> CachedTokenExampleProvider:
    provider = object.__new__(CachedTokenExampleProvider)
    provider._native_model = _NativeModel()  # ty: ignore[invalid-assignment]
    provider._sae = _ExecutorSAE()  # ty: ignore[invalid-assignment]
    provider._activation_type = object()  # ty: ignore[invalid-assignment]
    provider._layer = 1
    provider._hook_name = "blocks.1.hook_resid_post"
    provider._ignore_bos_features = True
    provider._ignore_bos_token_id = 2
    provider._batch_companion_token_strings = {
        0: "decoded-0",
        2: "decoded-2",
    }
    provider._batch_verified = False
    provider._batch_shadow_counter = 0
    provider._batch_validation_observations = []
    provider._batch_validation_anchor_target_keys = None
    provider._batch_gpu_thread_local = threading.local()
    provider._batch_gpu_streams_ready = False
    provider._batch_tolerance_profile = BatchToleranceProfile(
        name="cpu-native-executor-test",
        batch_size=4,
        selected_sae=NumericTolerance(
            max_abs=max_abs,
            p99_abs=max_abs,
            mean_abs=max_abs,
        ),
        raw_residual=NumericTolerance(
            max_abs=max_abs,
            p99_abs=max_abs,
            mean_abs=max_abs,
        ),
        calibration_contract=BatchCalibrationContract(
            model_name="fake-native-model",
            model_dtype="bfloat16",
            activation_type="residual",
            layer=1,
            segment_size=3,
            sae_release="fake-release",
            sae_id="fake-sae",
            hook_name="blocks.1.hook_resid_post",
        ),
        max_support_mismatches=0,
    )
    return provider


def _executor_prepared_batch() -> PreparedBatch:
    block = TargetBlock(
        ordinal=0,
        block_id=0,
        batch_size=4,
        targets_by_slot=(_target(0), _target(1), _target(2), None),
    )
    rows = (
        _prepared_doc(
            slot=0,
            feature_id=0,
            target_key="feature:0",
            model_token_ids=(2, 3, 4),
            real_token_ids=(3, 4),
            token_strings=("three", "four"),
            activation_slice_start=1,
        ),
        _prepared_doc(
            slot=1,
            feature_id=1,
            target_key="feature:1",
            model_token_ids=(2, 5),
            real_token_ids=(2, 5),
            token_strings=("bos", "five"),
            activation_slice_start=0,
        ),
        _prepared_doc(
            slot=2,
            feature_id=2,
            target_key="feature:2",
            model_token_ids=(2, 6, 2, 7),
            real_token_ids=(6, 2, 7),
            token_strings=("six", "bos", "seven"),
            activation_slice_start=1,
        ),
        None,
    )
    return PreparedBatch(block=block, rows=rows, model_width=4)


def _row_independent_native_activations(
    _model,
    tokens: t.Tensor,
    _layer: int,
    *,
    activation_type,
    hook_name,
) -> t.Tensor:
    assert activation_type is not None
    assert hook_name == "blocks.1.hook_resid_post"
    positions = t.arange(tokens.shape[1], dtype=t.float32).expand(tokens.shape[0], -1)
    return t.stack((tokens.to(t.float32), positions), dim=-1)


def test_native_block_projection_forwards_only_true_length_singletons_and_preserves_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    prepared = _executor_prepared_batch()
    seen_tokens: list[t.Tensor] = []

    def _recording_native(*args, **kwargs) -> t.Tensor:
        seen_tokens.append(args[1].detach().clone())
        return _row_independent_native_activations(*args, **kwargs)

    monkeypatch.setattr(token_examples_module, "_get_activations_native", _recording_native)

    selected, residuals = provider._run_native_projection_batch(
        prepared,
        capture_residual_slots=(0, 2),
    )

    assert len(seen_tokens) == 3
    assert sorted(tuple(tokens[0].tolist()) for tokens in seen_tokens) == [
        (2, 3, 4),
        (2, 5),
        (2, 6, 2, 7),
    ]
    assert all(tokens.ndim == 2 and tokens.shape[0] == 1 for tokens in seen_tokens)
    assert selected.dtype == t.float32 and selected.device.type == "cpu"
    assert selected.shape == (4, 4)
    assert selected[0].tolist() == [2.0, 3.0, 4.0, 0.0]
    assert selected[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert selected[2].tolist() == [2.0, 7.0, 4.0, 10.0]
    assert selected[3].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert set(residuals) == {0, 2}
    assert t.equal(
        residuals[2],
        t.tensor([[2.0, 0.0], [6.0, 1.0], [2.0, 2.0], [7.0, 3.0]]),
    )


@pytest.mark.parametrize("capture_slots", [(0, 0), (3,), (-1,), (4,)])
def test_native_batch_projection_rejects_duplicate_out_of_range_and_dummy_residual_slots(
    monkeypatch: pytest.MonkeyPatch,
    capture_slots: tuple[int, ...],
) -> None:
    provider = _executor_provider()
    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _row_independent_native_activations,
    )
    with pytest.raises(AutoInterpDataError, match="residual capture"):
        provider._run_native_projection_batch(
            _executor_prepared_batch(),
            capture_residual_slots=capture_slots,
        )


def test_native_batch_projection_rejects_wrong_model_activation_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        lambda *_args, **_kwargs: t.zeros((4, 3, 2), dtype=t.float32),
    )
    with pytest.raises(AutoInterpDataError, match="activation shape mismatch"):
        provider._run_native_projection_batch(_executor_prepared_batch())


def test_native_batch_projection_translates_oom_without_retry_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    calls: list[tuple[int, ...]] = []

    def _oom(_model, tokens, *_args, **_kwargs):
        calls.append(tuple(tokens[0].tolist()))
        raise t.cuda.OutOfMemoryError("injected")

    monkeypatch.setattr(token_examples_module, "_get_activations_native", _oom)
    with pytest.raises(
        AutoInterpDataError,
        match=(r"no smaller concurrency or serial fallback.*block=0 slot=0 model_length=3"),
    ):
        provider._run_native_projection_batch(_executor_prepared_batch())
    assert 1 <= len(calls) <= 3
    assert len(calls) == len(set(calls))
    assert set(calls) <= {(2, 3, 4), (2, 5), (2, 6, 2, 7)}


def test_cuda_oom_detection_traverses_wrapped_cause_context_and_cycles() -> None:
    direct = t.cuda.OutOfMemoryError("direct")
    cause_wrapper = RuntimeError("backend adapter")
    cause_wrapper.__cause__ = direct
    context_wrapper = RuntimeError("outer adapter")
    context_wrapper.__context__ = cause_wrapper
    assert _exception_chain_contains_cuda_oom(direct) is True
    assert _exception_chain_contains_cuda_oom(cause_wrapper) is True
    assert _exception_chain_contains_cuda_oom(context_wrapper) is True

    cycle_a = RuntimeError("a")
    cycle_b = RuntimeError("b")
    cycle_a.__context__ = cycle_b
    cycle_b.__context__ = cycle_a
    assert _exception_chain_contains_cuda_oom(cycle_a) is False
    assert _exception_chain_contains_cuda_oom(RuntimeError("ordinary")) is False


def test_native_projection_translates_backend_wrapped_cuda_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()

    def _wrapped_oom(*_args, **_kwargs):
        try:
            raise t.cuda.OutOfMemoryError("inner CUDA allocation")
        except t.cuda.OutOfMemoryError as exc:
            raise RuntimeError("native collector wrapped failure") from exc

    monkeypatch.setattr(token_examples_module, "_get_activations_native", _wrapped_oom)
    with pytest.raises(
        AutoInterpDataError,
        match=r"no smaller concurrency or serial fallback.*block=0 slot=0",
    ):
        provider._run_native_projection_batch(_executor_prepared_batch())


def test_native_projection_uses_exactly_two_persistent_workers_when_executor_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    prepared = _executor_prepared_batch()
    lock = threading.Lock()
    active = 0
    max_active = 0
    thread_names: list[str] = []

    def _instrumented_native(*args, **kwargs) -> t.Tensor:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            thread_names.append(threading.current_thread().name)
        time.sleep(0.02)
        try:
            return _row_independent_native_activations(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _instrumented_native,
    )
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="test-persistent-gpu",
    ) as executor:
        first, _ = provider._run_native_projection_batch(
            prepared,
            gpu_executor=executor,
        )
        second, _ = provider._run_native_projection_batch(
            prepared,
            gpu_executor=executor,
        )

    assert t.equal(first, second)
    assert max_active == 2
    assert len(thread_names) == 6
    assert set(thread_names) == {"test-persistent-gpu_0", "test-persistent-gpu_1"}


def test_native_projection_reports_lowest_slot_failure_even_when_later_slot_fails_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    slot_one_started = threading.Event()

    def _ordered_failures(*args, **kwargs) -> t.Tensor:
        tokens = tuple(args[1][0].tolist())
        if tokens == (2, 3, 4):
            assert slot_one_started.wait(timeout=2.0)
            raise RuntimeError("canonical-slot-zero-failure")
        if tokens == (2, 5):
            slot_one_started.set()
            raise RuntimeError("earlier-wall-clock-slot-one-failure")
        return _row_independent_native_activations(*args, **kwargs)

    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _ordered_failures,
    )
    with pytest.raises(RuntimeError, match="canonical-slot-zero-failure"):
        provider._run_native_projection_batch(_executor_prepared_batch())
    assert slot_one_started.is_set()


@pytest.mark.parametrize(
    ("capture_slots", "projected", "residual", "message"),
    [
        (
            (),
            t.zeros(3, dtype=t.float64),
            None,
            "projection contract mismatch",
        ),
        (
            (),
            t.zeros(2, dtype=t.float32),
            None,
            "projection contract mismatch",
        ),
        (
            (0,),
            t.zeros(3, dtype=t.float32),
            None,
            "omitted requested residual",
        ),
        (
            (0,),
            t.zeros(3, dtype=t.float32),
            t.zeros((2, 2), dtype=t.float32),
            "residual contract mismatch",
        ),
        (
            (),
            t.zeros(3, dtype=t.float32),
            t.zeros((3, 2), dtype=t.float32),
            "unrequested residual",
        ),
    ],
)
def test_native_projection_rejects_every_singleton_result_contract_violation(
    capture_slots: tuple[int, ...],
    projected: t.Tensor,
    residual: t.Tensor | None,
    message: str,
) -> None:
    provider = _executor_provider()

    def _invalid_singleton(
        _row: PreparedDoc,
        *,
        capture_residual: bool,
    ) -> tuple[t.Tensor, t.Tensor | None]:
        _ = capture_residual
        return projected, residual

    cast(Any, provider)._run_native_projection_singleton = _invalid_singleton
    with pytest.raises(AutoInterpDataError, match=message):
        provider._run_native_projection_batch(
            _executor_prepared_batch(),
            capture_residual_slots=capture_slots,
        )


def test_native_projection_drains_running_sibling_before_reporting_contract_failure() -> None:
    provider = _executor_provider()
    slot_one_started = threading.Event()
    slot_one_finished = threading.Event()

    def _invalid_first_singleton(
        row: PreparedDoc,
        *,
        capture_residual: bool,
    ) -> tuple[t.Tensor, t.Tensor | None]:
        assert capture_residual is False
        if row.slot == 0:
            assert slot_one_started.wait(timeout=2.0)
            return t.zeros(len(row.model_token_ids), dtype=t.float64), None
        if row.slot == 1:
            slot_one_started.set()
            time.sleep(0.05)
            slot_one_finished.set()
        return t.zeros(len(row.model_token_ids), dtype=t.float32), None

    cast(Any, provider)._run_native_projection_singleton = _invalid_first_singleton
    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(AutoInterpDataError, match="projection contract mismatch"):
            provider._run_native_projection_batch(
                _executor_prepared_batch(),
                gpu_executor=executor,
            )
        assert slot_one_finished.is_set()


def test_native_projection_drains_running_work_when_a_later_submission_fails() -> None:
    provider = _executor_provider()
    started = threading.Event()
    finished = threading.Event()

    def _slow_singleton(
        row: PreparedDoc,
        *,
        capture_residual: bool,
    ) -> tuple[t.Tensor, t.Tensor | None]:
        assert capture_residual is False
        started.set()
        time.sleep(0.05)
        finished.set()
        return t.zeros(len(row.model_token_ids), dtype=t.float32), None

    class _FailSecondSubmission:
        def __init__(self, executor: ThreadPoolExecutor) -> None:
            self.executor = executor
            self.calls = 0

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected second submission failure")
            future = self.executor.submit(*args, **kwargs)
            assert started.wait(timeout=2.0)
            return future

    cast(Any, provider)._run_native_projection_singleton = _slow_singleton
    with ThreadPoolExecutor(max_workers=1) as executor:
        failing_executor = _FailSecondSubmission(executor)
        with pytest.raises(RuntimeError, match="second submission failure"):
            provider._run_native_projection_batch(
                _executor_prepared_batch(),
                gpu_executor=cast(Any, failing_executor),
            )
        assert failing_executor.calls == 2
        assert finished.is_set()


def test_gpu_singleton_stream_publication_synchronizes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    cast(Any, provider._native_model).device = t.device("cuda:7")
    synchronized: list[t.device] = []
    monkeypatch.setattr(t.cuda, "synchronize", lambda device: synchronized.append(device))

    provider._synchronize_before_gpu_singleton_streams()
    provider._synchronize_before_gpu_singleton_streams()

    assert synchronized == [t.device("cuda:7")]
    assert provider._batch_gpu_streams_ready is True


def test_native_and_serial_projection_reject_missing_loaded_dependencies() -> None:
    provider = _executor_provider()
    provider._native_model = None
    with pytest.raises(AutoInterpDataError, match="native batch projection dependencies"):
        provider._run_native_projection_batch(_executor_prepared_batch())
    with pytest.raises(AutoInterpDataError, match="stream setup requires a loaded native model"):
        provider._synchronize_before_gpu_singleton_streams()
    with pytest.raises(AutoInterpDataError, match="singleton projection dependencies"):
        provider._run_native_projection_singleton(
            cast(PreparedDoc, _executor_prepared_batch().rows[0]),
            capture_residual=False,
        )
    with pytest.raises(AutoInterpDataError, match="serial batch-validation dependencies"):
        provider._serial_surfaces_for_prepared_doc(
            cast(PreparedDoc, _executor_prepared_batch().rows[0]),
            capture_residual=True,
        )


def test_real_activation_slice_drops_only_injected_bos_and_zeros_all_natural_bos() -> None:
    provider = _executor_provider()
    injected = _prepared_doc(
        model_token_ids=(2, 10, 2, 12),
        real_token_ids=(10, 2, 12),
        token_strings=("ten", "bos", "twelve"),
        activation_slice_start=1,
    )
    natural = _prepared_doc(
        model_token_ids=(2, 10, 2),
        real_token_ids=(2, 10, 2),
        token_strings=("bos", "ten", "bos"),
        activation_slice_start=0,
    )
    projected = t.tensor([100.0, 1.0, 2.0, 3.0], dtype=t.float32)
    assert provider._real_activation_slice(projected, injected).tolist() == [1.0, 0.0, 3.0]
    assert provider._real_activation_slice(projected[:3], natural).tolist() == [0.0, 1.0, 0.0]
    residual = t.arange(8, dtype=t.float32).reshape(4, 2)
    assert t.equal(provider._real_residual_slice(residual, injected), residual[1:4])


def test_real_activation_slice_requires_resolved_bos_when_zeroing_is_enabled() -> None:
    provider = _executor_provider()
    provider._ignore_bos_token_id = None
    with pytest.raises(AutoInterpDataError, match="requires a resolved BOS id"):
        provider._real_activation_slice(t.ones(4, dtype=t.float32), _prepared_doc())


def test_real_activation_and_residual_slices_reject_shape_length_and_nonfinite() -> None:
    provider = _executor_provider()
    row = _prepared_doc()
    with pytest.raises(AutoInterpDataError, match="cannot satisfy prepared slice"):
        provider._real_activation_slice(t.zeros((2, 2)), row)
    with pytest.raises(AutoInterpDataError, match="cannot satisfy prepared slice"):
        provider._real_activation_slice(t.zeros(2), row)
    with pytest.raises(AutoInterpDataError, match="cannot satisfy prepared slice"):
        provider._real_residual_slice(t.zeros((2, 2)), row)
    bad = t.zeros((4, 2), dtype=t.float32)
    bad[2, 0] = float("nan")
    with pytest.raises(AutoInterpDataError, match="NaN/Inf"):
        provider._real_residual_slice(bad, row)


def test_full_startup_gate_checks_repeat_serial_residual_projection_and_companions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    prepared = _executor_prepared_batch()
    calls = 0

    def _native(*args, **kwargs) -> t.Tensor:
        nonlocal calls
        calls += 1
        return _row_independent_native_activations(*args, **kwargs)

    monkeypatch.setattr(token_examples_module, "_get_activations_native", _native)

    docs = provider._execute_prepared_batch(prepared)

    # Candidate + repeat + companion each execute all three singleton rows;
    # startup serial checks execute all three probes, and the periodic shadow
    # executes one additional serial singleton.
    assert calls == 13
    assert provider._batch_verified is True
    assert provider._batch_shadow_counter == 1
    assert docs[3] is None
    assert docs[0] is not None and docs[0].token_ids == (3, 4)
    assert docs[1] is not None and docs[1].sae_acts[:, 0].tolist() == [0.0, 1.0]
    assert docs[2] is not None and docs[2].sae_acts[:, 0].tolist() == [7.0, 0.0, 10.0]


def test_startup_gate_rejects_non_repeat_exact_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider(max_abs=100.0)
    prepared = _executor_prepared_batch()
    calls_by_tokens: dict[tuple[int, ...], int] = {}

    def _drifting(*args, **kwargs) -> t.Tensor:
        token_key = tuple(args[1][0].tolist())
        calls_by_tokens[token_key] = calls_by_tokens.get(token_key, 0) + 1
        result = _row_independent_native_activations(*args, **kwargs)
        if token_key == (2, 3, 4) and calls_by_tokens[token_key] == 2:
            result = result.clone()
            result[0, 0, 0] += 1.0
        return result

    monkeypatch.setattr(token_examples_module, "_get_activations_native", _drifting)
    with pytest.raises(AutoInterpBatchValidationError, match="candidate is not repeat-exact"):
        provider._execute_prepared_batch(prepared)
    assert provider._batch_verified is False


def test_startup_gate_rejects_missing_residual_capture_slots() -> None:
    provider = _executor_provider(max_abs=100.0)

    def _missing_residuals(
        prepared: PreparedBatch,
        *,
        capture_residual_slots: tuple[int, ...] = (),
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        _ = capture_residual_slots, gpu_executor
        return t.zeros(
            (prepared.block.batch_size, prepared.model_width),
            dtype=t.float32,
        ), {}

    cast(Any, provider)._run_native_projection_batch = _missing_residuals
    with pytest.raises(AutoInterpDataError, match="wrong probe slots"):
        provider._execute_prepared_batch(_executor_prepared_batch())


def test_startup_companion_probe_must_hold_probe_fixed_and_change_another_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _row_independent_native_activations,
    )
    row0 = _prepared_doc(
        slot=0,
        feature_id=0,
        target_key="feature:0",
        model_token_ids=(2, 3, 4),
        real_token_ids=(3, 4),
        token_strings=("three", "four"),
        activation_slice_start=1,
    )
    row1 = replace(row0, slot=1, feature_id=1, target_key="feature:1")
    prepared = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), _target(1), None, None),
        ),
        rows=(row0, row1, None, None),
        model_width=4,
    )

    cast(Any, provider)._same_probe_companion_batch = lambda _prepared, *, probe_slot: prepared
    with pytest.raises(AutoInterpDataError, match="did not change any companion"):
        provider._execute_prepared_batch(prepared)
    assert provider._batch_verified is False

    changed_probe = PreparedBatch(
        block=prepared.block,
        rows=(
            replace(
                row0,
                model_token_ids=(2, 9, 4),
                real_token_ids=(9, 4),
                token_strings=("nine", "four"),
            ),
            replace(row1, model_token_ids=(2, 8, 4), real_token_ids=(8, 4)),
            None,
            None,
        ),
        model_width=4,
    )
    cast(Any, provider)._same_probe_companion_batch = lambda _prepared, *, probe_slot: changed_probe
    with pytest.raises(AutoInterpDataError, match="changed the probe row"):
        provider._execute_prepared_batch(prepared)
    assert provider._batch_verified is False


def test_calibration_shadow_rejects_missing_candidate_residual() -> None:
    provider = _executor_provider(max_abs=100.0)
    provider._batch_verified = True
    provider._batch_validation_anchor_target_keys = frozenset({"feature:0"})
    prepared = _executor_prepared_batch()

    def _missing_residuals(
        prepared: PreparedBatch,
        *,
        capture_residual_slots: tuple[int, ...] = (),
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        _ = capture_residual_slots, gpu_executor
        return t.zeros(
            (prepared.block.batch_size, prepared.model_width),
            dtype=t.float32,
        ), {}

    cast(Any, provider)._run_native_projection_batch = _missing_residuals
    cast(Any, provider)._serial_surfaces_for_prepared_doc = lambda row, *, capture_residual: (
        t.zeros(len(row.real_token_ids), dtype=t.float32),
        t.zeros((len(row.real_token_ids), 2), dtype=t.float32) if capture_residual else None,
    )
    with pytest.raises(AutoInterpDataError, match="calibration residual shadow is missing"):
        provider._execute_prepared_batch(prepared)


def test_executor_rejects_empty_batch_and_nonfinite_selected_surface() -> None:
    provider = _executor_provider(max_abs=100.0)
    prepared = _executor_prepared_batch()
    empty = PreparedBatch(
        block=prepared.block,
        rows=(None,) * prepared.block.batch_size,
        model_width=prepared.model_width,
    )
    assert provider._startup_probe_slots(empty) == ()
    with pytest.raises(AutoInterpDataError, match="empty prepared batch"):
        provider._execute_prepared_batch(empty)

    provider._batch_verified = True
    provider._batch_shadow_counter = 4

    def _nonfinite(
        prepared: PreparedBatch,
        *,
        capture_residual_slots: tuple[int, ...] = (),
        gpu_executor: ThreadPoolExecutor | None = None,
    ) -> tuple[t.Tensor, dict[int, t.Tensor]]:
        _ = capture_residual_slots, gpu_executor
        candidate = t.zeros(
            (prepared.block.batch_size, prepared.model_width),
            dtype=t.float32,
        )
        candidate[0, 1] = float("nan")
        return candidate, {}

    cast(Any, provider)._run_native_projection_batch = _nonfinite
    with pytest.raises(AutoInterpDataError, match="SAE activations contain NaN/Inf"):
        provider._execute_prepared_batch(prepared)


def test_prepare_doc_and_companion_probe_reject_missing_surfaces() -> None:
    provider = _executor_provider()
    prepared = _executor_prepared_batch()
    provider._tokenizer = None
    with pytest.raises(AutoInterpDataError, match="requires a loaded tokenizer"):
        provider._prepare_doc(
            block=prepared.block,
            slot=0,
            target=_target(0),
            phase=MaterializationPhase.positive,
            doc_ref=_doc_ref(0),
        )

    provider._tokenizer = object()
    cast(Any, provider)._full_text = lambda _ref: "text"
    cast(Any, provider)._reforward_surface = lambda _text, _ref: ([], [], 0)
    with pytest.raises(AutoInterpDataError, match="produced no SAE tokens"):
        provider._prepare_doc(
            block=prepared.block,
            slot=0,
            target=_target(0),
            phase=MaterializationPhase.positive,
            doc_ref=_doc_ref(0),
        )
    with pytest.raises(AutoInterpDataError, match="companion probe slot 3 is empty"):
        provider._same_probe_companion_batch(prepared, probe_slot=3)
    provider._sae = None
    with pytest.raises(AutoInterpDataError, match="requires a loaded SAE"):
        provider._same_probe_companion_batch(prepared, probe_slot=0)


def test_prepare_doc_freezes_exact_text_token_decode_and_injected_bos_surface() -> None:
    provider = _executor_provider()
    block = _executor_prepared_batch().block
    doc_ref = _doc_ref(17, start=32, end=34)
    decode_calls: list[tuple[int, ...]] = []

    class _Tokenizer:
        @staticmethod
        def decode(token_ids: list[int]) -> str:
            decode_calls.append(tuple(token_ids))
            return {7: " seven", 8: "eight"}[token_ids[0]]

    provider._tokenizer = _Tokenizer()
    cast(Any, provider)._full_text = lambda ref: "careful π text"
    cast(Any, provider)._reforward_surface = lambda text, ref: ([2, 7, 8], [7, 8], 32)

    row = provider._prepare_doc(
        block=block,
        slot=0,
        target=_target(0),
        phase=MaterializationPhase.positive,
        doc_ref=doc_ref,
    )

    assert row == PreparedDoc(
        block_ordinal=block.ordinal,
        slot=0,
        target_key="feature:0",
        feature_id=0,
        phase=MaterializationPhase.positive,
        doc_ref=doc_ref,
        source_text_hash=hashlib.sha256("careful π text".encode()).hexdigest(),
        model_token_ids=(2, 7, 8),
        real_token_ids=(7, 8),
        token_strings=(" seven", "eight"),
        token_offset=32,
        activation_slice_start=1,
    )
    assert decode_calls == [(7,), (8,)]


def test_single_real_row_companion_probe_populates_a_second_valid_feature_slot() -> None:
    provider = _executor_provider()
    row = _prepared_doc(
        slot=0,
        feature_id=0,
        target_key="feature:0",
        model_token_ids=(2, 3, 4),
        real_token_ids=(3, 4),
        token_strings=("three", "four"),
        activation_slice_start=1,
    )
    original = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), None, None, None),
        ),
        rows=(row, None, None, None),
        model_width=4,
    )

    companion = provider._same_probe_companion_batch(original, probe_slot=0)

    assert original.rows == (row, None, None, None)
    assert original.block.targets_by_slot == (_target(0), None, None, None)
    assert companion.block.targets_by_slot[1] == ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(1,),
        target_key="__batch_companion__:1",
        display_name="batch_companion_1",
    )
    assert companion.rows[0] == row
    assert companion.rows[1] is not None
    assert companion.rows[1].feature_id == 1
    assert t.equal(companion.token_tensor()[0], companion.token_tensor()[1])
    assert companion.rows[2:] == (None, None)


def test_identical_real_companions_are_explicitly_perturbed_without_changing_probe() -> None:
    provider = _executor_provider()
    row0 = _prepared_doc(
        slot=0,
        feature_id=0,
        target_key="feature:0",
        model_token_ids=(2, 3, 4),
        real_token_ids=(3, 4),
        token_strings=("three", "four"),
        activation_slice_start=1,
    )
    row1 = replace(row0, slot=1, feature_id=1, target_key="feature:1")
    prepared = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), _target(1), None, None),
        ),
        rows=(row0, row1, None, None),
        model_width=4,
    )

    companion = provider._same_probe_companion_batch(prepared, probe_slot=0)

    assert companion.rows[0] == row0
    assert companion.rows[1] is not None
    assert companion.rows[1].model_token_ids == (0, 3, 4)
    assert companion.rows[1].real_token_ids == row1.real_token_ids
    assert companion.rows[1].token_strings == row1.token_strings
    assert t.equal(companion.token_tensor()[0], prepared.token_tensor()[0])
    assert not t.equal(companion.token_tensor()[1], prepared.token_tensor()[1])


def test_natural_bos_companion_uses_predecoded_token_without_tokenizer_access() -> None:
    provider = _executor_provider()

    class _TokenizerMustRemainIdle:
        @staticmethod
        def decode(_token_ids: list[int]) -> str:
            raise AssertionError("companion probe must not access the tokenizer")

    provider._tokenizer = _TokenizerMustRemainIdle()
    row0 = _prepared_doc(
        slot=0,
        feature_id=0,
        target_key="feature:0",
        model_token_ids=(2, 3),
        real_token_ids=(2, 3),
        token_strings=("bos", "three"),
        activation_slice_start=0,
    )
    row1 = replace(row0, slot=1, feature_id=1, target_key="feature:1")
    prepared = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), _target(1), None, None),
        ),
        rows=(row0, row1, None, None),
        model_width=4,
    )

    companion = provider._same_probe_companion_batch(prepared, probe_slot=0)

    assert companion.rows[1] is not None
    assert companion.rows[1].model_token_ids == (0, 3)
    assert companion.rows[1].real_token_ids == (0, 3)
    assert companion.rows[1].token_strings == ("decoded-0", "three")


def test_companion_perturbation_rejects_missing_distinct_or_predecoded_token() -> None:
    provider = _executor_provider()
    row0 = _prepared_doc(
        slot=0,
        feature_id=0,
        target_key="feature:0",
        model_token_ids=(0, 3),
        real_token_ids=(3,),
        token_strings=("three",),
        activation_slice_start=1,
    )
    row1 = replace(row0, slot=1, feature_id=1, target_key="feature:1")
    prepared = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), _target(1), None, None),
        ),
        rows=(row0, row1, None, None),
        model_width=4,
    )
    provider._ignore_bos_token_id = None
    with pytest.raises(AutoInterpDataError, match="distinct valid token ID"):
        provider._same_probe_companion_batch(prepared, probe_slot=0)

    provider._ignore_bos_token_id = 2
    natural0 = replace(
        row0,
        model_token_ids=(0, 3),
        real_token_ids=(0, 3),
        token_strings=("zero", "three"),
        activation_slice_start=0,
    )
    natural1 = replace(natural0, slot=1, feature_id=1, target_key="feature:1")
    natural = PreparedBatch(
        block=prepared.block,
        rows=(natural0, natural1, None, None),
        model_width=4,
    )
    provider._batch_companion_token_strings = {0: "decoded-0"}
    with pytest.raises(AutoInterpDataError, match="no predecoded string.*token ID 2"):
        provider._same_probe_companion_batch(natural, probe_slot=0)


def test_companion_probe_predecodes_all_replacement_strings_before_prefetch() -> None:
    provider = _executor_provider()
    decode_calls: list[tuple[int, ...]] = []

    class _Tokenizer:
        @staticmethod
        def decode(token_ids: list[int]) -> str:
            decode_calls.append(tuple(token_ids))
            return f"token-{token_ids[0]}"

    provider._tokenizer = _Tokenizer()
    provider._batch_companion_token_strings = {}

    provider._prepare_batch_companion_token_strings()

    assert decode_calls == [(0,), (2,)]
    assert provider._batch_companion_token_strings == {
        0: "token-0",
        2: "token-2",
    }


@pytest.mark.parametrize(
    ("tokenizer", "bos_token_id", "message"),
    [
        (None, 2, "loaded tokenizer and BOS id"),
        (object(), None, "loaded tokenizer and BOS id"),
        (object(), 0, "collides with pad/dummy"),
    ],
)
def test_companion_probe_predecode_rejects_missing_or_colliding_contract(
    tokenizer: object | None,
    bos_token_id: int | None,
    message: str,
) -> None:
    provider = _executor_provider()
    provider._tokenizer = tokenizer
    provider._ignore_bos_token_id = bos_token_id

    with pytest.raises(AutoInterpDataError, match=message):
        provider._prepare_batch_companion_token_strings()


def test_single_real_row_companion_probe_fails_if_sae_has_no_second_valid_slot() -> None:
    provider = _executor_provider()
    cast(Any, provider._sae).W_enc = t.nn.Parameter(t.zeros((2, 1), dtype=t.float32))
    row = _prepared_doc(slot=0, feature_id=0, target_key="feature:0")
    prepared = PreparedBatch(
        block=TargetBlock(
            ordinal=0,
            block_id=0,
            batch_size=4,
            targets_by_slot=(_target(0), None, None, None),
        ),
        rows=(row, None, None, None),
        model_width=4,
    )

    with pytest.raises(AutoInterpDataError, match="cannot populate a second valid"):
        provider._same_probe_companion_batch(prepared, probe_slot=0)


def test_calibration_anchor_mode_shadows_every_anchor_row_in_each_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _executor_provider()
    provider._batch_validation_anchor_target_keys = frozenset({"feature:0", "feature:2"})
    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _row_independent_native_activations,
    )

    provider._execute_prepared_batch(_executor_prepared_batch())

    periodic = [
        observation
        for observation in provider._batch_validation_observations
        if observation["surface"] == "selected_sae_periodic_shadow"
    ]
    assert len(periodic) == 2
    assert [observation["target_key"] for observation in periodic] == [
        "feature:0",
        "feature:2",
    ]
    assert "slot=0" in cast(str, periodic[0]["context"])
    assert "slot=2" in cast(str, periodic[1]["context"])
    residual = [
        observation
        for observation in provider._batch_validation_observations
        if observation["surface"] == "raw_residual_calibration_shadow"
    ]
    assert len(residual) == 2
    assert [observation["target_key"] for observation in residual] == [
        "feature:0",
        "feature:2",
    ]
    assert [observation["surface_kind"] for observation in residual] == [
        "raw_residual",
        "raw_residual",
    ]
    assert all(observation["require_support_bound"] is False for observation in residual)

    provider._execute_prepared_batch(_executor_prepared_batch())
    residual = [
        observation
        for observation in provider._batch_validation_observations
        if observation["surface"] == "raw_residual_calibration_shadow"
    ]
    assert len(residual) == 4


def test_batch_validation_observation_rejects_empty_target_key() -> None:
    provider = _executor_provider()
    with pytest.raises(BeartypeCallHintParamViolation, match="target_key"):
        provider._validate_batch_surface(
            t.zeros(2, dtype=t.float32),
            t.zeros(2, dtype=t.float32),
            target_key="",
            context="invalid target",
            surface="selected_sae_test",
            surface_kind=BatchSurfaceKind.SELECTED_SAE,
        )


@pytest.mark.parametrize(
    "bad",
    [set(), frozenset(), {"feature:0"}, frozenset({""}), frozenset({1})],
)
def test_provider_rejects_invalid_internal_calibration_anchor_contract(
    tmp_path: Path,
    bad,
) -> None:
    cfg = AutoInterpFilterConfig(
        feature_dir="unused",
        output_dir=str(tmp_path),
        token_examples=TokenExamplesConfig(materialization_batch_size=PositiveInt.parse(128)),
    )
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="batch_validation_anchor_target_keys",
    ):
        CachedTokenExampleProvider(
            cfg,
            feature_dir=Path(cfg.feature_dir),
            n_sae_features=128,
            batch_validation_anchor_target_keys=bad,
        )


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"materialization_prefetch": 1}, "materialization_prefetch"),
        ({"batch_pair_timing_sink": ()}, "batch_pair_timing_sink"),
        (
            {"allow_unvalidated_batch_profile": 1},
            "allow_unvalidated_batch_profile",
        ),
    ],
)
def test_provider_rejects_invalid_internal_scheduling_instrumentation(
    tmp_path: Path,
    kwargs: dict[str, Any],
    field_name: str,
) -> None:
    cfg = AutoInterpFilterConfig(feature_dir="unused", output_dir=str(tmp_path))
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        CachedTokenExampleProvider(
            cfg,
            feature_dir=Path(cfg.feature_dir),
            n_sae_features=128,
            **kwargs,
        )


def test_prepare_rejects_missing_internal_calibration_anchor_before_file_io(
    tmp_path: Path,
) -> None:
    cfg = AutoInterpFilterConfig(feature_dir="missing", output_dir=str(tmp_path))
    timing_sink: list[dict[str, object]] = [{"stale": True}]
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=128,
        batch_validation_anchor_target_keys=frozenset({"feature:1"}),
        batch_pair_timing_sink=timing_sink,
    )
    with pytest.raises(AutoInterpDataError, match="anchors are absent"):
        provider.prepare([_target(0)])
    assert timing_sink == []


def test_prepare_rejects_invalid_full_and_scoring_target_relationships_before_file_io(
    tmp_path: Path,
) -> None:
    cfg = AutoInterpFilterConfig(feature_dir="missing", output_dir=str(tmp_path))
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=Path(cfg.feature_dir),
        n_sae_features=128,
    )
    with pytest.raises(AutoInterpDataError, match="full target universe contains duplicates"):
        provider.prepare([_target(0), replace(_target(1), target_key="feature:0")])
    with pytest.raises(AutoInterpDataError, match="not an exact member"):
        provider.prepare([_target(0)], scoring_targets=[_target(1)])
    with pytest.raises(AutoInterpDataError, match="scoring target subset contains duplicate"):
        provider.prepare([_target(0)], scoring_targets=[_target(0), _target(0)])
