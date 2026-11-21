"""Cross-site pin for the canonical signed class-margin sign rule.

The rule "positive -> avoid-rich, negative -> rich-avoid (sign flip)" is
expressed at several independent sites across axbench_bpb_adv (scoring,
aggregation, the GCG objective's desired/contrast ordering) and the sibling
axbench_bpb aggregation. They MUST all agree, and in particular the GCG search
objective ``g`` (the unweighted steering margin) must equal ``primary_margin_y``
-- the identity the whole design rests on. This test fails loudly if any single
site diverges from the shared ``signed_class_margin`` helper.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_bpb.bpb_aggregation import compute_per_tuple_rollup
from steering.eval.axbench_bpb_adv.aggregation import _signed_margin
from steering.eval.axbench_bpb_adv.optimize import objective_value

_MODES = ["positive", "negative"]
_RICH = [-1.5, 0.0, 1.0, 4.25]
_AVOID = [-2.0, 0.0, 2.5, 7.0]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("rich", _RICH)
@pytest.mark.parametrize("avoid", _AVOID)
def test_aggregation_signed_margin_matches_shared_helper(
    mode: str, rich: float, avoid: float
) -> None:
    assert _signed_margin(
        mode=mode, mean_rich=rich, mean_avoid=avoid
    ) == pytest.approx(signed_class_margin(mode=mode, rich=rich, avoid=avoid))


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("rich", _RICH)
@pytest.mark.parametrize("avoid", _AVOID)
def test_objective_value_equals_primary_margin(
    mode: str, rich: float, avoid: float
) -> None:
    # Mode -> (desired, contrast), mirroring model_optimize._evaluate exactly:
    # positive -> desired=rich, contrast=avoid; negative -> desired=avoid,
    # contrast=rich.
    if mode == "positive":
        desired, contrast = rich, avoid
    else:
        desired, contrast = avoid, rich
    g = objective_value(
        loss_desired=desired,
        loss_contrast=contrast,
    )
    assert g == pytest.approx(signed_class_margin(mode=mode, rich=rich, avoid=avoid))


def _bpb_score_row(*, sample_id: str, content_class: str, bpb: float, mode: str) -> dict:
    return {
        "method_name": "m",
        "concept_id": 1,
        "mode": mode,
        "content_class": content_class,
        "target": "encdec",
        "coefficient_str": "0",
        "sample_id": sample_id,
        "bpb_intervened": bpb,
        "n_output_tokens": 10,
        "n_output_bytes": 40,
        "truncated": False,
    }


def test_bpb_aggregation_primary_margin_matches_shared_helper() -> None:
    rich_bpb = 1.0
    avoid_bpb = 3.0
    scores = pa.Table.from_pylist(
        [
            _bpb_score_row(
                sample_id="r_pos", content_class="concept_rich", bpb=rich_bpb, mode="positive"
            ),
            _bpb_score_row(
                sample_id="a_pos", content_class="concept_avoiding", bpb=avoid_bpb, mode="positive"
            ),
            _bpb_score_row(
                sample_id="r_neg", content_class="concept_rich", bpb=rich_bpb, mode="negative"
            ),
            _bpb_score_row(
                sample_id="a_neg", content_class="concept_avoiding", bpb=avoid_bpb, mode="negative"
            ),
        ]
    )
    rows = compute_per_tuple_rollup(scores).to_pylist()
    by_mode = {row["mode"]: row for row in rows}
    assert set(by_mode) == {"positive", "negative"}
    for mode, row in by_mode.items():
        assert row["primary_margin"] == pytest.approx(
            signed_class_margin(mode=mode, rich=rich_bpb, avoid=avoid_bpb)
        )
