"""Unit tests for loss-mask construction."""
from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.feature_mask import FeatureMaskMode
from steering.eval.axbench_bpb.teacher_force import (
    accumulate_bpb,
    assert_enc_ids_match_cluster_contract,
)
from steering.eval.axbench_bpb.teacher_force import (
    build_output_only_mask as _typed_build_output_only_mask,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.refined_types import PositiveInt

# Reuses the fixture builder from test_teacher_force_encdec_inputs.py rather
# than re-implementing FeatureClusterConsolidation.build/FeatureMask/
# FeatureVocab wiring a second time (owner lens: duplicate-helper hunt).
from tests.axbench_bpb.unit.test_teacher_force_encdec_inputs import (
    _cluster_contract,
)

# Two genuinely-constructible, differently-SIZED compacted checkpoint
# vocabs (V=3 of 4 vs. V=999 of 1000, both excluding exactly id 3) -- NOT
# "compacted vs. uncompacted": review follow-up to task-f1b-report.md's
# original framing, which used a fabricated full-width/identity
# checkpoint.feature_vocab that is UNCONSTRUCTIBLE for any real, validated
# ClusterFeatureContract (see test_teacher_force_encdec_inputs.py ::
# TestClusterContract.test_checkpoint_feature_vocab_none_raises_before_
# packing's docstring for the full invariant). assert_enc_ids_match_
# cluster_contract never reads checkpoint.feature_vocab at all, so these
# two regimes exist purely to make that independence an explicit, checked
# fact (varying an attribute the function must not consult, in two
# concretely different but always-legal shapes) rather than an unverified
# assumption.
_NARROW_VOCAB_KWARGS: dict[str, Any] = {}
_WIDE_VOCAB_KWARGS: dict[str, Any] = {
    "n_features": 1000, "mask_ids": (3,), "mask_mode": FeatureMaskMode.blacklist,
}


def build_output_only_mask(*, n_full_prefix: int, n_output: int) -> torch.Tensor:
    """Build the refined scalar inputs owned by these behavioral tests."""
    return _typed_build_output_only_mask(
        n_full_prefix=PositiveInt.parse(n_full_prefix),
        n_output=PositiveInt.parse(n_output),
    )


def test_output_only_mask_shape_and_content():
    """n_full_prefix=3, n_output=2 → T=5; target indices [0..3];
    mask should be [F, F, T, T] with len 4."""
    mask = build_output_only_mask(n_full_prefix=3, n_output=2)
    assert mask.dtype == torch.bool
    assert mask.shape == (4,)
    assert mask.tolist() == [False, False, True, True]


_INVALID_POSITIVE_INT = st.one_of(
    st.integers(max_value=0),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)


@settings(max_examples=500, derandomize=True)
@given(
    st.one_of(
        st.tuples(st.just("n_full_prefix"), _INVALID_POSITIVE_INT),
        st.tuples(st.just("n_output"), _INVALID_POSITIVE_INT),
    )
)
def test_output_only_mask_rejects_every_invalid_length(
    invalid: tuple[str, Any],
) -> None:
    field, value = invalid
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        _typed_build_output_only_mask(
            n_full_prefix=(
                value if field == "n_full_prefix" else PositiveInt.parse(3)
            ),
            n_output=value if field == "n_output" else PositiveInt.parse(2),
        )


def test_output_only_mask_single_prefix_single_output():
    """n_full_prefix=1, n_output=1 → T=2; mask has len 1 → [True]."""
    mask = build_output_only_mask(n_full_prefix=1, n_output=1)
    assert mask.tolist() == [True]


def test_output_only_mask_all_prefix_long():
    """n_full_prefix=5, n_output=1 → T=6; mask [F,F,F,F,T], len 5."""
    mask = build_output_only_mask(n_full_prefix=5, n_output=1)
    assert mask.tolist() == [False, False, False, False, True]


def test_accumulate_bpb_matches_spec_formula():
    """Byte-exact: nats = sum(loss * byte_weight over output positions);
    bpb = nats / (ln2 * total_bytes). Test with loss=[0.5, 1.0, 2.0, 3.0],
    mask=[F,F,T,T], byte_weights_output=[3, 4] → nats=18.0, bytes=7."""
    loss = torch.tensor([0.5, 1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    bpb, n_bytes = accumulate_bpb(loss, mask, byte_weights_output)
    expected = 18.0 / (math.log(2) * 7)
    assert abs(bpb - expected) < 1e-6
    assert n_bytes == 7


def test_accumulate_bpb_ignores_non_mask_losses():
    """Out-of-mask losses DO NOT contribute to nats; sentinel values outside
    the mask must be benign. Put giant values at prefix positions; verify
    result is unaffected."""
    loss = torch.tensor([1e9, -1e9, 2.0, 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    bpb, n_bytes = accumulate_bpb(loss, mask, byte_weights_output)
    expected = 18.0 / (math.log(2) * 7)
    assert abs(bpb - expected) < 1e-6
    assert n_bytes == 7


def test_accumulate_bpb_non_finite_outside_mask_ok():
    """Non-finite losses at non-mask positions are ignored — the check
    runs only over in_mask_losses."""
    loss = torch.tensor([0.5, float("nan"), 2.0, 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    # Should not raise.
    bpb, n_bytes = accumulate_bpb(loss, mask, byte_weights_output)
    expected = 18.0 / (math.log(2) * 7)
    assert abs(bpb - expected) < 1e-6


def test_accumulate_bpb_non_finite_inside_mask_raises():
    """Non-finite loss at a masked position raises AxBenchBpbError."""
    loss = torch.tensor([0.5, 1.0, float("nan"), 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    with pytest.raises(AxBenchBpbError, match="non-finite"):
        accumulate_bpb(loss, mask, byte_weights_output)


def test_accumulate_bpb_inf_inside_mask_raises():
    """Also flags inf, not just NaN."""
    loss = torch.tensor([0.5, 1.0, float("inf"), 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    with pytest.raises(AxBenchBpbError, match="non-finite"):
        accumulate_bpb(loss, mask, byte_weights_output)


def test_accumulate_bpb_shape_mismatch_raises():
    loss = torch.tensor([0.5, 1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, True])  # different shape
    byte_weights_output = torch.tensor([3, 4], dtype=torch.int64)
    with pytest.raises(AxBenchBpbError, match="shape mismatch"):
        accumulate_bpb(loss, mask, byte_weights_output)


def test_accumulate_bpb_byte_weights_length_mismatch_raises():
    """byte_weights_output length must match mask True count."""
    loss = torch.tensor([0.5, 1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, True, True])  # 2 True positions
    byte_weights_output = torch.tensor([3, 4, 5], dtype=torch.int64)  # 3 entries
    with pytest.raises(AxBenchBpbError, match="must match"):
        accumulate_bpb(loss, mask, byte_weights_output)


def test_accumulate_bpb_zero_bytes_raises():
    """All-zero byte weights → TypeError-like domain failure."""
    loss = torch.tensor([0.5, 1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, True, True])
    byte_weights_output = torch.tensor([0, 0], dtype=torch.int64)
    with pytest.raises(AxBenchBpbError, match="zero bytes"):
        accumulate_bpb(loss, mask, byte_weights_output)


def test_accumulate_bpb_single_mask_position():
    """Minimal case: 1 output position, 1-byte output."""
    loss = torch.tensor([0.5, 1.0])
    mask = torch.tensor([False, True])
    byte_weights_output = torch.tensor([1], dtype=torch.int64)
    bpb, n_bytes = accumulate_bpb(loss, mask, byte_weights_output)
    expected = 1.0 / (math.log(2) * 1)
    assert abs(bpb - expected) < 1e-6
    assert n_bytes == 1


# ---------------------------------------------------------------------------
# assert_enc_ids_match_cluster_contract (PR #739 Track F1b): the pre-forward
# backstop wired immediately before every clustered encdec forward. Unlike
# _build_encdec_seg_map_and_encoder's checkpoint-vocab-exact-match packing
# check (tests/axbench_bpb/unit/test_teacher_force_encdec_inputs.py ::
# TestClusterContract), this function reads ONLY cluster_contract.
# consolidation / .feature_mask -- it never touches
# cluster_contract.checkpoint.feature_vocab -- so it is unconditionally
# vocab-SIZE-agnostic (see _NARROW_VOCAB_KWARGS/_WIDE_VOCAB_KWARGS above).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vocab_kwargs",
    [_NARROW_VOCAB_KWARGS, _WIDE_VOCAB_KWARGS],
    ids=["narrow_vocab", "wide_vocab"],
)
def test_assert_enc_ids_match_cluster_contract_accepts_representative_only(
    vocab_kwargs: dict[str, Any],
) -> None:
    contract = _cluster_contract(**vocab_kwargs)
    enc_feature_ids = torch.tensor([[[0, 2]]], dtype=torch.int64)
    enc_mask = torch.tensor([[[True, True]]], dtype=torch.bool)

    assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)  # must not raise


@pytest.mark.parametrize(
    "vocab_kwargs",
    [_NARROW_VOCAB_KWARGS, _WIDE_VOCAB_KWARGS],
    ids=["narrow_vocab", "wide_vocab"],
)
def test_assert_enc_ids_match_cluster_contract_rejects_consolidated_away_id(
    vocab_kwargs: dict[str, Any],
) -> None:
    """1 is consolidated into representative 0 by the default cluster
    ``(0, 1)`` fixture; a packed id of 1 in a REAL (unmasked) slot must
    raise, regardless of the checkpoint's persisted vocabulary SIZE."""
    contract = _cluster_contract(**vocab_kwargs)
    enc_feature_ids = torch.tensor([[[1]]], dtype=torch.int64)
    enc_mask = torch.tensor([[[True]]], dtype=torch.bool)

    with pytest.raises(AxBenchBpbError, match="pre-forward packed IDs"):
        assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)


@pytest.mark.parametrize(
    "vocab_kwargs",
    [_NARROW_VOCAB_KWARGS, _WIDE_VOCAB_KWARGS],
    ids=["narrow_vocab", "wide_vocab"],
)
def test_assert_enc_ids_match_cluster_contract_rejects_masked_id(
    vocab_kwargs: dict[str, Any],
) -> None:
    """id 3 is mask-disallowed in both regimes (narrow: outside the default
    whitelist ``(0, 1, 2)``; wide: explicitly blacklisted)."""
    contract = _cluster_contract(**vocab_kwargs)
    enc_feature_ids = torch.tensor([[[3]]], dtype=torch.int64)
    enc_mask = torch.tensor([[[True]]], dtype=torch.bool)

    with pytest.raises(AxBenchBpbError, match="pre-forward packed IDs"):
        assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)


def test_assert_enc_ids_match_cluster_contract_ignores_padding_slots() -> None:
    """A padding (enc_mask=False) slot holding a would-be-rejected id (the
    consolidated-away 1) must be silently ignored -- mirrors
    assert_enc_ids_in_vocab's real-slots-only contract."""
    contract = _cluster_contract()
    enc_feature_ids = torch.tensor([[[0, 1]]], dtype=torch.int64)
    enc_mask = torch.tensor([[[True, False]]], dtype=torch.bool)

    assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)  # must not raise


def test_assert_enc_ids_match_cluster_contract_no_real_slots_does_not_raise() -> None:
    contract = _cluster_contract()
    enc_feature_ids = torch.zeros((1, 1, 3), dtype=torch.int64)
    enc_mask = torch.zeros((1, 1, 3), dtype=torch.bool)

    assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)  # must not raise


def test_assert_enc_ids_match_cluster_contract_mask_shifted_representative() -> None:
    """The same mask-shifted-representative fixture as
    TestClusterContract.test_mask_shifted_representative, exercised through
    this standalone pre-forward assert instead of full packing: the shifted
    representative (0) is accepted, its consolidated-away sibling (2's
    excluded slot maps survivors to 0) is not directly testable here since
    2 itself is mask-disallowed -- the representative-vs-sibling distinction
    is covered by test_mask_shifted_representative; this proves the
    standalone assert agrees with the packing-level checks on the accept
    side using the identical fixture."""
    contract = _cluster_contract(mask_ids=(0, 1, 3), clusters=((2, 0, 1),))
    enc_feature_ids = torch.tensor([[[0]]], dtype=torch.int64)
    enc_mask = torch.tensor([[[True]]], dtype=torch.bool)

    assert_enc_ids_match_cluster_contract(enc_feature_ids, enc_mask, contract)  # must not raise
