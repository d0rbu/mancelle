"""#1249: the value that carries a feature bag into encdec direction fitting.

`EncoderConditioning` exists so the bag travels WITH its identity: the digest is
stamped into the direction artifact, and a direction fit under one bag can never
be silently reused against another. These tests pin the invariants that make
that stamp trustworthy -- sortedness (digest is order-independent), rejection of
empty bags (an empty bag IS the unconditioned forward #1249 removed), and exact
tensor agreement with the eval-side encoder-input builder.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.eval.axbench_base.baselines.conditioning import (
    EncoderConditioning,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def test_entries_must_be_sorted_so_the_digest_is_order_independent() -> None:
    with pytest.raises(AxBenchBaseError, match="sorted"):
        EncoderConditioning(entries=((5, 1), (2, 0)))


def test_duplicate_feature_ids_are_rejected() -> None:
    with pytest.raises(AxBenchBaseError, match="duplicate"):
        EncoderConditioning(entries=((2, 0), (2, 1)))


@pytest.mark.parametrize("bad", [[(1, 0)], ((1,),), ((1, "0"),)])
def test_malformed_entries_are_rejected(bad: object) -> None:
    with pytest.raises(AxBenchBaseError):
        EncoderConditioning(entries=bad)  # ty: ignore[invalid-argument-type]


def test_digest_is_stable_and_content_bound() -> None:
    a = EncoderConditioning(entries=((2, 0), (5, 1)))
    b = EncoderConditioning(entries=((2, 0), (5, 1)))
    c = EncoderConditioning(entries=((2, 0), (5, 0)))
    assert a.bag_digest == b.bag_digest, "equal bags must share a digest"
    assert a.bag_digest != c.bag_digest, "a bucket flip must change the digest"
    assert len(a.bag_digest) == 64


def test_as_tensors_matches_the_eval_side_builder_exactly() -> None:
    """The fit forward must see EXACTLY what the eval forward would see.

    Any divergence here means the direction is fit under different encoder
    inputs than the steering it will counter -- the #1249 defect in miniature.
    """
    entries = ((3, 1), (7, 0), (11, 1))
    cond = EncoderConditioning(entries=entries)
    got_fids, got_mags, got_mask = cond.as_tensors("cpu")
    want_fids, want_mags, want_mask = build_segmented_feature_encoder_inputs(
        [{3: 1, 7: 0, 11: 1}], "cpu"
    )
    assert t.equal(got_fids, want_fids)
    assert t.equal(got_mags, want_mags)
    assert t.equal(got_mask, want_mask)
    assert got_fids.shape[1] == 1, "one capture forward = one encoder segment"


def test_empty_bag_cannot_produce_tensors() -> None:
    """An empty bag is the removed defect, not a degenerate valid input."""
    cond = EncoderConditioning(entries=())
    with pytest.raises(AxBenchBaseError, match="empty"):
        cond.as_tensors("cpu")
