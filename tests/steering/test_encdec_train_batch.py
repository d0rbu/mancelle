"""Slice-10 audit: ``EncDecTrainBatch`` typed dataclass round-trip.

Covers plan T-5 + T-10, plus the slice-10 follow-up cleanup that REMOVED
the legacy 9-tuple branch from :meth:`EncDecTrainBatch.from_tuple`.

Coverage matrix:

* Happy paths
  * 10-tuple → typed view, all 10 fields populated.
  * 10-tuple with ``fp_targets=None`` → typed view preserves None.
  * 10-list (vs tuple) → typed view (lenient sequence type).
  * Already-typed ``EncDecTrainBatch`` → passthrough (idempotent).

* Length boundaries (slice-10 cleanup regression guards)
  * 9-tuple → ValueError (the removed branch).
  * 8 / 11 / empty / extreme → ValueError.

* Type contract
  * Non-sequence inputs (int / float / str / bytes / dict / set / None)
    → ValueError.

* Field assignment ordering
  * Pin that seq[i] maps to the i-th declared field, in declared order.
  * Pin that ``state_dict`` may be ``None`` or ``dict`` (the contract is
    not strictly enforced at construction; this matches existing producer
    behavior).

* Property tests (Hypothesis, derandomize=True, max_examples=200)
  * ANY length other than 10 is rejected.
  * Field passthrough: random sentinels at each of the 10 positions land
    in the matching field.
  * Round-trip: ``from_tuple(b.to_tuple()) == b`` for arbitrary inputs
    (we add a ``to_tuple`` helper on the test side to exercise the
    inverse — the dataclass itself does not need a ``to_tuple`` method).

* Integration
  * Real ``precomputed_data_loader`` yields 10-tuples (regression for the
    production path).
  * Yielded tuples round-trip through ``from_tuple`` cleanly.

See manifest §1.12, slice-10 plan F-5, slice-10 follow-up directive.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder.dataloader import EncDecTrainBatch
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    SEQ_LEN,
    make_precomputed_dataloader,
)

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


_EXPECTED_FIELDS = (
    "inputs",
    "targets",
    "enc_fids",
    "enc_mags",
    "enc_mask",
    "enc_pcts",
    "seg_map",
    "seg_doc_id",
    "fp_targets",
    "state_dict",
)


def _make_tensors(B=2, T=8, n_seg=3, F=4):
    """Build the tensor portion of a synthetic batch."""
    return {
        "inputs": torch.randint(0, 100, (B, T), dtype=torch.long),
        "targets": torch.randint(-1, 100, (B, T), dtype=torch.long),
        "enc_fids": torch.randint(0, 200, (B, n_seg, F), dtype=torch.long),
        "enc_mags": torch.randint(0, 4, (B, n_seg, F), dtype=torch.long),
        "enc_mask": torch.ones(B, n_seg, F, dtype=torch.bool),
        "enc_pcts": torch.rand(B, n_seg, F),
        "seg_map": torch.zeros(B, T, dtype=torch.long),
        "seg_doc_id": torch.zeros(B, n_seg, dtype=torch.long),
    }


def _build_10_tuple(
    fp: torch.Tensor | None = None, state: dict | None = None
):
    """Construct a synthetic 10-tuple for happy-path tests."""
    tens = _make_tensors()
    if state is None:
        state = {"epoch": 1}
    return (
        tens["inputs"],
        tens["targets"],
        tens["enc_fids"],
        tens["enc_mags"],
        tens["enc_mask"],
        tens["enc_pcts"],
        tens["seg_map"],
        tens["seg_doc_id"],
        fp,
        state,
    )


def _batch_to_tuple(b: EncDecTrainBatch) -> tuple:
    """Test-side inverse of from_tuple.

    Lives in the test module (not the dataclass) so the production code
    surface stays minimal — the inverse is only needed by the round-trip
    property test.
    """
    return tuple(getattr(b, name) for name in _EXPECTED_FIELDS)


# ---------------------------------------------------------------------------
# Happy-path unit tests
# ---------------------------------------------------------------------------


class TestFromTupleHappyPath:
    """Tests for the documented accepted input shapes."""

    def test_from_tuple_accepts_10_item_tuple(self):
        """10-tuple unpacks to all 10 fields with fp_targets present."""
        tens = _make_tensors()
        fp = torch.zeros(2, 3, 50)
        state = {"epoch": 1}
        raw = (
            tens["inputs"], tens["targets"], tens["enc_fids"], tens["enc_mags"],
            tens["enc_mask"], tens["enc_pcts"], tens["seg_map"], tens["seg_doc_id"],
            fp, state,
        )
        b = EncDecTrainBatch.from_tuple(raw)
        assert torch.equal(b.inputs, tens["inputs"])
        assert torch.equal(b.targets, tens["targets"])
        assert torch.equal(b.enc_fids, tens["enc_fids"])
        assert torch.equal(b.enc_mags, tens["enc_mags"])
        assert torch.equal(b.enc_mask, tens["enc_mask"])
        assert torch.equal(b.enc_pcts, tens["enc_pcts"])
        assert torch.equal(b.seg_map, tens["seg_map"])
        assert torch.equal(b.seg_doc_id, tens["seg_doc_id"])
        assert b.fp_targets is fp
        assert b.state_dict is state

    def test_from_tuple_accepts_10_item_list(self):
        """A list (vs tuple) of 10 items also works — lenient sequence type."""
        tens = _make_tensors()
        state = {"epoch": 1}
        raw_list = [
            tens["inputs"], tens["targets"], tens["enc_fids"], tens["enc_mags"],
            tens["enc_mask"], tens["enc_pcts"], tens["seg_map"], tens["seg_doc_id"],
            None, state,
        ]
        b = EncDecTrainBatch.from_tuple(raw_list)
        assert b.fp_targets is None
        assert b.state_dict is state

    def test_from_tuple_passes_through_existing_dataclass(self):
        """Already-typed batch → returned unchanged (idempotent)."""
        tens = _make_tensors()
        state = {"epoch": 1}
        raw_b = EncDecTrainBatch(
            inputs=tens["inputs"],
            targets=tens["targets"],
            enc_fids=tens["enc_fids"],
            enc_mags=tens["enc_mags"],
            enc_mask=tens["enc_mask"],
            enc_pcts=tens["enc_pcts"],
            seg_map=tens["seg_map"],
            seg_doc_id=tens["seg_doc_id"],
            fp_targets=None,
            state_dict=state,
        )
        out = EncDecTrainBatch.from_tuple(raw_b)
        assert out is raw_b

    def test_from_tuple_fp_targets_can_be_none_in_10_tuple(self):
        """fp_targets=None in a 10-tuple is a valid 10-tuple, NOT a 9-tuple.

        Crucial separation post-cleanup: the slot for fp_targets is always
        present in the production tuple even when ``n_sae_features`` is
        unset; the value is just ``None``. Producers MUST emit a 10-tuple
        with ``None`` rather than dropping the slot.
        """
        raw = _build_10_tuple(fp=None)
        b = EncDecTrainBatch.from_tuple(raw)
        assert b.fp_targets is None
        assert isinstance(b.state_dict, dict)

    def test_from_tuple_fp_targets_can_be_tensor_in_10_tuple(self):
        """fp_targets=tensor in a 10-tuple is also valid."""
        fp = torch.zeros(2, 3, 50)
        raw = _build_10_tuple(fp=fp)
        b = EncDecTrainBatch.from_tuple(raw)
        assert b.fp_targets is fp
        assert torch.equal(b.fp_targets, torch.zeros(2, 3, 50))


# ---------------------------------------------------------------------------
# Length-boundary error path tests (post-cleanup regression guards)
# ---------------------------------------------------------------------------


class TestFromTupleLengthBoundaries:
    """The 9-tuple branch was REMOVED in the slice-10 follow-up. These tests
    pin that any length other than 10 raises ValueError citing the count.
    """

    def test_from_tuple_rejects_9_item_tuple(self):
        """9-tuple → ValueError (the removed legacy branch).

        This is the headline regression guard for the cleanup. A producer
        that still emits a 9-tuple (e.g., a stale checkpoint loader) must
        surface as a clear error, not be silently fixed up with
        ``fp_targets=None`` as the old code did.
        """
        raw = tuple(range(9))
        with pytest.raises(ValueError) as excinfo:
            EncDecTrainBatch.from_tuple(raw)
        msg = str(excinfo.value)
        assert "10-item" in msg
        assert "9" in msg

    def test_from_tuple_rejects_8_item_tuple(self):
        """8-tuple → ValueError (one short of the legacy branch boundary)."""
        with pytest.raises(ValueError, match=r"10-item.*got 8 items"):
            EncDecTrainBatch.from_tuple(tuple(range(8)))

    def test_from_tuple_rejects_11_item_tuple(self):
        """11-tuple → ValueError (one past the accepted boundary)."""
        with pytest.raises(ValueError, match=r"10-item.*got 11 items"):
            EncDecTrainBatch.from_tuple(tuple(range(11)))

    def test_from_tuple_rejects_empty_tuple(self):
        """0-tuple → ValueError (degenerate edge)."""
        with pytest.raises(ValueError, match=r"10-item.*got 0 items"):
            EncDecTrainBatch.from_tuple(())

    def test_from_tuple_rejects_empty_list(self):
        """Empty list → ValueError (degenerate edge, list path)."""
        with pytest.raises(ValueError, match=r"10-item.*got 0 items"):
            EncDecTrainBatch.from_tuple([])

    def test_from_tuple_rejects_singleton(self):
        """1-tuple → ValueError (a common bug shape)."""
        with pytest.raises(ValueError, match=r"10-item.*got 1 items"):
            EncDecTrainBatch.from_tuple((42,))

    @pytest.mark.parametrize("bad_len", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 20])
    def test_from_tuple_rejects_all_non_10_lengths(self, bad_len):
        """Parametrized sweep across non-10 lengths."""
        raw = tuple(range(bad_len))
        with pytest.raises(ValueError) as excinfo:
            EncDecTrainBatch.from_tuple(raw)
        assert str(bad_len) in str(excinfo.value)
        assert "10-item" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Type-contract error path tests
# ---------------------------------------------------------------------------


class TestFromTupleTypeContract:
    """Inputs that are not tuple / list / EncDecTrainBatch are rejected."""

    @pytest.mark.parametrize(
        "bad",
        [42, 3.14, "ten characters", b"bytes", {"a": 1}, set(), None],
        ids=["int", "float", "str", "bytes", "dict", "set", "None"],
    )
    def test_from_tuple_rejects_non_iterable(self, bad):
        """Non-sequence inputs raise ValueError mentioning the bad type."""
        with pytest.raises(ValueError) as excinfo:
            EncDecTrainBatch.from_tuple(bad)
        msg = str(excinfo.value).lower()
        # Either the error mentions 'tuple', 'list', or the bad type's name.
        assert any(k in msg for k in ("tuple", "list", "encdec"))

    def test_from_tuple_rejects_dict_explicitly(self):
        """Dict — even one with 10 keys — is rejected as a non-sequence."""
        bad = {f"k{i}": i for i in range(10)}
        with pytest.raises(ValueError) as excinfo:
            EncDecTrainBatch.from_tuple(bad)  # ty: ignore[invalid-argument-type]
        # Dict is rejected at the type-check, not the length-check; so the
        # message should be about the type, not the count.
        assert "dict" in str(excinfo.value).lower() or "tuple" in str(excinfo.value).lower()

    def test_from_tuple_rejects_generator(self):
        """A generator (iterable but not list/tuple) is rejected.

        Production code never passes generators here (the dataloader yields
        concrete tuples), so we lock in the conservative type-check.
        """

        def _gen():
            yield from range(10)

        with pytest.raises(ValueError):
            EncDecTrainBatch.from_tuple(_gen())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Field-assignment ordering and dataclass invariants
# ---------------------------------------------------------------------------


class TestDataclassInvariants:
    """Lock in the dataclass shape so consumers can rely on positions."""

    def test_field_order_matches_yield_tuple(self):
        """The dataclass field order must match the yield tuple order so
        that a future direct-construction site (slice 12) can pass
        positional args without re-ordering. Pin the contract.
        """
        got = tuple(f.name for f in fields(EncDecTrainBatch))
        assert got == _EXPECTED_FIELDS

    def test_dataclass_is_frozen(self):
        """Frozen contract: cannot mutate after construction.

        Frozen dataclasses are required so consumers can't accidentally
        swap a tensor mid-iteration (e.g., the prefetch wrapper would get
        confused if a fid tensor turned into a mask tensor).
        """
        tens = _make_tensors()
        b = EncDecTrainBatch(
            inputs=tens["inputs"],
            targets=tens["targets"],
            enc_fids=tens["enc_fids"],
            enc_mags=tens["enc_mags"],
            enc_mask=tens["enc_mask"],
            enc_pcts=tens["enc_pcts"],
            seg_map=tens["seg_map"],
            seg_doc_id=tens["seg_doc_id"],
            fp_targets=None,
            state_dict={},
        )
        with pytest.raises((AttributeError, TypeError)):
            b.inputs = tens["targets"]  # ty: ignore[invalid-assignment]

    def test_from_tuple_field_assignment_order(self):
        """Each seq[i] is assigned to the i-th declared field.

        Uses string sentinels so any swap is detectable. Bypasses runtime
        type checking on tensor fields by passing strings — ``from_tuple``
        does not validate field types, only positions and arity.
        """
        sentinels = [f"v{i}" for i in range(10)]
        b = EncDecTrainBatch.from_tuple(tuple(sentinels))  # type: ignore[arg-type]
        for idx, field_name in enumerate(_EXPECTED_FIELDS):
            assert getattr(b, field_name) == f"v{idx}", (
                f"field {field_name} got {getattr(b, field_name)!r}, "
                f"expected v{idx}"
            )

    def test_from_tuple_state_dict_slot_holds_none(self):
        """The state_dict slot accepts ``None`` and arbitrary mappings.

        ``from_tuple`` does NOT enforce that the slot is a dict — that
        would over-constrain a documented "opaque resume bookkeeping"
        contract. We only pin the position here.

        (See report from this audit re: behavioral surprises — type
        enforcement on slots is not implemented; consumers are
        responsible for validating ``state_dict`` shape if they care.)
        """
        raw = (None,) * 9 + (None,)
        b = EncDecTrainBatch.from_tuple(raw)
        assert b.state_dict is None

    def test_from_tuple_state_dict_slot_holds_dict(self):
        """And of course a dict in the state_dict slot is the common case."""
        d = {"epoch": 5, "shard": "x"}
        raw = (None,) * 9 + (d,)
        b = EncDecTrainBatch.from_tuple(raw)
        assert b.state_dict is d

    def test_error_message_format_includes_actual_length(self):
        """The error message must include the received length so callers
        can debug at-a-glance.
        """
        for n in (0, 1, 5, 9, 11, 100):
            with pytest.raises(ValueError) as excinfo:
                EncDecTrainBatch.from_tuple(tuple(range(n)))
            assert str(n) in str(excinfo.value), (
                f"length {n} missing from error message: "
                f"{excinfo.value}"
            )


# ---------------------------------------------------------------------------
# Property tests (Hypothesis)
# ---------------------------------------------------------------------------


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(bad_len=st.integers(min_value=0, max_value=20).filter(lambda n: n != 10))
def test_property_any_seq_length_other_than_10_rejected(bad_len: int):
    """Hypothesis: for any length in [0, 20] other than 10, ValueError."""
    raw = tuple(range(bad_len))
    with pytest.raises(ValueError) as excinfo:
        EncDecTrainBatch.from_tuple(raw)
    assert str(bad_len) in str(excinfo.value)
    assert "10-item" in str(excinfo.value)


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    payload=st.lists(
        st.one_of(
            st.integers(),
            st.text(),
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False),
        ),
        min_size=10,
        max_size=10,
    ),
)
def test_property_field_passthrough(payload: list[Any]):
    """Hypothesis: 10 random sentinels round-trip through the field positions.

    For any 10-element sequence, ``from_tuple(seq).field_i == seq[i]`` for
    each declared field. This is the positional contract.
    """
    b = EncDecTrainBatch.from_tuple(tuple(payload))  # type: ignore[arg-type]
    for idx, field_name in enumerate(_EXPECTED_FIELDS):
        assert getattr(b, field_name) == payload[idx]


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    payload=st.lists(
        st.one_of(st.integers(), st.text(), st.none()),
        min_size=10,
        max_size=10,
    ),
)
def test_property_round_trip_via_to_tuple(payload: list[Any]):
    """Hypothesis: round-trip via the test-side ``_batch_to_tuple`` inverse.

    Build a typed batch, decompose it back into a tuple, rebuild from
    that tuple — the two typed views must compare equal field-by-field.

    (We don't put a ``to_tuple`` on the dataclass itself because nothing
    in production needs it yet; this property only exercises the
    positional invariant on the test side.)
    """
    b1 = EncDecTrainBatch.from_tuple(tuple(payload))  # type: ignore[arg-type]
    b2 = EncDecTrainBatch.from_tuple(_batch_to_tuple(b1))
    for field_name in _EXPECTED_FIELDS:
        assert getattr(b1, field_name) == getattr(b2, field_name)


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(bad=st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.binary(),
    st.dictionaries(st.text(), st.integers(), max_size=10),
    st.sets(st.integers(), max_size=10),
    st.none(),
))
def test_property_non_sequence_inputs_rejected(bad: Any):
    """Hypothesis: any non-tuple/list/EncDecTrainBatch input is rejected."""
    with pytest.raises(ValueError):
        EncDecTrainBatch.from_tuple(bad)


# ---------------------------------------------------------------------------
# Integration tests — real dataloader produces 10-tuples
# ---------------------------------------------------------------------------


class TestIntegrationWithRealDataloader:
    """End-to-end checks that the production path stays in sync with the
    typed view contract.

    The real ``precomputed_data_loader`` is what feeds training and
    evaluation. If a future refactor accidentally drops the ``fp_targets``
    slot and emits a 9-tuple, these tests fail fast.
    """

    def test_dataloader_yields_10_tuples(self, tmp_path):
        """Production dataloader yields exactly 10-tuples (not 9)."""
        dl = make_precomputed_dataloader(tmp_path, B=2)
        batch = next(dl)
        assert isinstance(batch, tuple)
        assert len(batch) == 10, (
            f"Production dataloader emitted {len(batch)}-tuple; the typed "
            f"view requires 10. Did a refactor drop fp_targets?"
        )

    def test_dataloader_to_from_tuple_round_trip(self, tmp_path):
        """The yielded tuple feeds cleanly into ``from_tuple``."""
        dl = make_precomputed_dataloader(tmp_path, B=2)
        batch = next(dl)
        b = EncDecTrainBatch.from_tuple(batch)
        # Tensor fields are non-None.
        assert b.inputs is not None
        assert b.targets is not None
        assert b.enc_fids is not None
        assert b.enc_mags is not None
        assert b.enc_mask is not None
        assert b.enc_pcts is not None
        assert b.seg_map is not None
        assert b.seg_doc_id is not None
        # state_dict is always a dict from the production producer.
        assert isinstance(b.state_dict, dict)

    def test_dataloader_yields_10_tuple_with_fp_targets_when_n_sae_set(
        self, tmp_path
    ):
        """When ``n_sae_features`` is supplied, fp_targets is a tensor.

        Pins that ``None`` is not silently emitted in the fp_targets slot
        when the config is set; the slot still exists either way (10-tuple
        in both cases).
        """
        dl = make_precomputed_dataloader(
            tmp_path, B=2, n_sae_features=N_SAE_FEATURES
        )
        batch = next(dl)
        assert len(batch) == 10
        b = EncDecTrainBatch.from_tuple(batch)
        assert b.fp_targets is not None
        assert b.fp_targets.dtype == torch.float32

    def test_dataloader_input_shapes_match_seq_len(self, tmp_path):
        """Sanity: dataloader honors SEQ_LEN."""
        dl = make_precomputed_dataloader(tmp_path, B=3)
        b = EncDecTrainBatch.from_tuple(next(dl))
        assert b.inputs.shape == (3, SEQ_LEN)
        assert b.targets.shape == (3, SEQ_LEN)
