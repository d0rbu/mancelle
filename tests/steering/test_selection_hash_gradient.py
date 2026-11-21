"""I1 + sensitivity tests for the gradient block in compute_selection_hash.

The gradient block is canonicalized CENTRALLY inside compute_selection_hash
(all four ANTI-DESYNC sites pass the raw feature_selection dict, so they
inherit identical semantics with zero per-site edits and the cross-site AST
detector's expected kwarg set is unchanged).
"""

from __future__ import annotations

from typing import Any

import pytest

from steering.gradient_overlay import (
    GRAD_SCORE_REDUCTION,
    GRAD_SPLIT_POLICY_VERSION,
)
from steering.packed_batch_io import compute_selection_hash

# The stock conditional_encdec_train.yaml feature_selection block. This hash
# is FROZEN: it was computed at the pre-gradient HEAD and verified byte-equal
# through the central canonicalization (gradient key absent => the payload
# never changes). If this test ever fails, every existing sel/<hash>/ shard
# on disk has been orphaned — do NOT update the constant without a migration
# story (invariant I1).
_STOCK_SELECTION: dict[str, Any] = {
    "distribution": "geometric",
    "mean": 5.0,
    "std": 5.0,
    "min_k": 1,
    "max_k": 25,
    "delta_weights": "absolute_percentile",
    "mode": "top_k_percentile",
}
_I1_FROZEN_GOLDEN = "f350f1c958b15166c53e36e89b8ca598c9194b4497a09c8b2d21c686bf330397"


def _hash(positive: dict, **kw) -> str:
    return compute_selection_hash(
        positive_selection=positive,
        negative_selection=kw.pop("negative_selection", None),
        delta_config=kw.pop("delta_config", None),
        num_buckets=kw.pop("num_buckets", 2),
        seed=kw.pop("seed", 42),
        **kw,
    )


def _grad_selection(**grad) -> dict:
    sel = dict(_STOCK_SELECTION)
    del sel["delta_weights"]  # incompatible with gradient ranking
    sel["gradient"] = {"mode": "future_half_split", **grad}
    return sel


class TestI1ByteIdentity:
    def test_frozen_golden(self):
        assert _hash(dict(_STOCK_SELECTION)) == _I1_FROZEN_GOLDEN

    def test_explicit_null_gradient_is_byte_identical(self):
        with_null = dict(_STOCK_SELECTION)
        with_null["gradient"] = None
        assert _hash(with_null) == _I1_FROZEN_GOLDEN

    def test_input_dict_not_mutated(self):
        sel = _grad_selection()
        before = dict(sel)
        _hash(sel)
        assert sel == before, "compute_selection_hash must not mutate its input"


class TestGradientSensitivity:
    def test_gradient_changes_hash(self):
        assert _hash(_grad_selection()) != _I1_FROZEN_GOLDEN

    def test_modes_distinct(self):
        hashes = {
            _hash(_grad_selection(mode=m))
            for m in ("future_half_split", "future_self_full", "self")
        }
        assert len(hashes) == 3

    def test_min_final_segment_tokens_distinct(self):
        assert _hash(_grad_selection(min_final_segment_tokens=8)) != _hash(
            _grad_selection(min_final_segment_tokens=4)
        )

    def test_model_dtype_distinct(self):
        assert _hash(_grad_selection(model_dtype="bfloat16")) != _hash(
            _grad_selection(model_dtype="float32")
        )

    def test_canonicalization_fills_defaults(self):
        """An explicit-default block hashes identically to an implicit one
        (the canonical form pins mode/model_dtype/min_final + the
        code-versioned reduction/split-policy constants)."""
        implicit = _hash(_grad_selection())
        explicit = _hash(
            _grad_selection(model_dtype="bfloat16", min_final_segment_tokens=8)
        )
        assert implicit == explicit
        # The version constants are part of the payload: bumping them MUST
        # re-address sel dirs. (Guard the current values so a bump is a
        # conscious, test-visible decision.)
        assert GRAD_SCORE_REDUCTION == "pct_relu_neg_max"  # pct * sensitivity
        # v2: single-segment docs score unmerged; one-token final units are
        # enumerated zero-signal (see GRAD_SPLIT_POLICY_VERSION history). The
        # score-reduction switch does NOT touch the schedule, so this stays 2.
        assert GRAD_SPLIT_POLICY_VERSION == 2

    def test_determinism(self):
        assert _hash(_grad_selection()) == _hash(_grad_selection())

    def test_malformed_gradient_raises(self):
        sel = dict(_STOCK_SELECTION)
        sel["gradient"] = {"mode": "sideways"}
        with pytest.raises(ValueError, match="Unsupported gradient"):
            _hash(sel)
        sel["gradient"] = {"typo": 1}
        with pytest.raises(ValueError):
            _hash(sel)
