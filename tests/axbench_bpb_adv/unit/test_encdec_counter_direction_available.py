"""A clustered source must be able to counter-steer its encdec targets.

The gap this pins cost a full 3.5 h sweep. `axbench_base` trained scalar
baselines for `decoder_only` ONLY when a cluster contract was present:

    targets=(("decoder_only",) if state.cluster_contract is not None
             else ("decoder_only", "encdec"))

But counter-steering loads the counter direction in the BASELINE'S OWN model
coordinates -- "the encdec direction for an encdec baseline, NOT the
decoder-only one" (intervention_build) -- because the hook is installed on that
model and a direction from another model is a different residual basis, not a
rescaling. So an encdec-target row at non-zero strength needs
`<counter_direction_family>/encdec/`, which a clustered run never produced.

Why it hid for so long: the requirement is gated on `adversarial_strength != 0`,
and magnitude-0 returns the base intervention byte-identical. Every clustered run
before the first real counter-steering sweep used `magnitudes: [0.0]`.
"""

from __future__ import annotations

import inspect

import scripts.eval.axbench_base_eval as base_eval
from steering.eval.axbench_bpb_adv.config import CounterDirectionFamily


def _training_targets_source(stage_name: str) -> str:
    src = inspect.getsource(base_eval._run_single_stage)
    marker = f"Stage.{stage_name}"
    start = src.index(marker)
    # Bound the slice at the stage's own `return`, not by character count: a
    # fixed window silently truncates when the block gains a comment, which
    # makes this test pass or fail on prose rather than on code.
    end = src.index("\n        return\n", start)
    return src[start:end]


def test_both_scalar_baseline_stages_train_the_encdec_target() -> None:
    """Neither stage may restrict targets by clustered-ness.

    A `cluster_contract`-conditional target list is the exact defect: it makes
    encdec counter-steering impossible on precisely the runs that use encdec
    steering methods.
    """
    for stage_name in ("TRAIN_DIFFMEAN", "TRAIN_LINEAR_PROBE"):
        body = _training_targets_source(stage_name)
        assert '"encdec"' in body, (
            f"{stage_name} must train the encdec target; without it an encdec "
            "row at non-zero adversarial_strength has no counter direction"
        )
        assert "cluster_contract is not None" not in body, (
            f"{stage_name} must not gate its target list on clustered-ness -- "
            "that is the #1245-adjacent defect this test exists to prevent"
        )


def test_every_counter_direction_family_is_covered_for_encdec() -> None:
    """Fixing only diffmean would leave the other setting silently broken.

    `counter_direction_family` is a two-valued Literal, and the counter path
    resolves whichever is configured. A source that trains only diffmean/encdec
    fails identically -- and just as invisibly -- under
    `counter_direction_family: linear_probe`.
    """
    families = set(CounterDirectionFamily.__args__)
    assert families == {"diffmean", "linear_probe"}, (
        f"unexpected counter-direction families {families}; this test enumerates "
        "the stages that must train each one"
    )
    stage_by_family = {
        "diffmean": "TRAIN_DIFFMEAN",
        "linear_probe": "TRAIN_LINEAR_PROBE",
    }
    for family in families:
        stage = stage_by_family[family]
        body = _training_targets_source(stage)
        assert '"encdec"' in body, (
            f"counter_direction_family={family!r} is selectable, so its stage "
            f"({stage}) must train encdec-target directions"
        )
        # Presence of the literal is NOT sufficient: the pre-fix code contained
        # "encdec" inside the `else` arm of a clustered conditional, so it was
        # written but never REACHED on a clustered run. Asserting only presence
        # made this test pass against the exact defect it names.
        assert "cluster_contract is not None" not in body, (
            f"{stage} reaches the encdec target only on the unclustered arm; "
            f"counter_direction_family={family!r} would still have no encdec "
            "direction on a clustered source"
        )
