from __future__ import annotations

import pytest

from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    replace_single_token,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_bpb_adv.editable_tokens import (
    replace_editable_tokens,
    select_final_k_editable_span,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def test_select_final_k_editable_span_rejects_invalid_or_bos_spans() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="k must be"):
        select_final_k_editable_span([0, 1], k=-1)
    with pytest.raises(AxBenchBpbAdvError, match="BOS"):
        select_final_k_editable_span([], k=0)
    with pytest.raises(AxBenchBpbAdvError, match="fewer than"):
        select_final_k_editable_span([0, 1], k=2)

def test_replace_editable_tokens_rejects_length_and_boundary_violations() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="after BOS"):
        replace_editable_tokens([0, 1], EditableSpan(start=0, count=1), [2])
    with pytest.raises(AxBenchBpbAdvError, match="count"):
        replace_editable_tokens([0, 1], EditableSpan(start=1, count=-1), [])
    with pytest.raises(AxBenchBpbAdvError, match="exceeds"):
        replace_editable_tokens([0, 1], EditableSpan(start=1, count=2), [2, 3])
    with pytest.raises(AxBenchBpbAdvError, match="replacement length"):
        replace_editable_tokens([0, 1, 2], EditableSpan(start=1, count=2), [9])

    assert replace_editable_tokens([0, 1, 2], EditableSpan(start=1, count=2), [3, 4]) == [
        0,
        3,
        4,
    ]


def test_replace_single_token_rejects_bos_and_out_of_range_positions() -> None:
    # replace_single_token moved to feature_gcg.edit_primitives and raises FeatureGcgError
    with pytest.raises(FeatureGcgError, match="BOS"):
        replace_single_token([0, 1], position=0, token_id=2)
    with pytest.raises(FeatureGcgError, match="out of range"):
        replace_single_token([0, 1], position=2, token_id=2)

    assert replace_single_token([0, 1], position=1, token_id=3) == [0, 3]
