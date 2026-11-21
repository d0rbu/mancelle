"""Load the committed feature-DiffMean bags as encdec fit conditioning (#1249).

Stage 3a fits DiffMean/LinearProbe directions for the ``encdec`` target. Those
forwards must be conditioned by the concept's feature bag: before #1249 they
used an all-masked bag, which the code itself described as "effectively a
decoder-only residual trajectory on the encdec backbone" -- the direction
described activations the model never produces when steered, and the #1228
scale guard caught it as a negative ``max_activation`` (-6404.02, concept 2).

WHICH BAG: the committed fair feature-DiffMean bag itself
---------------------------------------------------------
This module does NOT recompute anything. The attack prefixes' completions ARE
the stage-1b/1c synth samples ("the synth pipeline emits N rich and M avoid
samples per prefix" -- attack_rows.py), so the eval-time bag and any fit-time
reconstruction would be the same constructor over the same corpus. Rather than
rebuild it modulo the fair split, we load the committed artifact the eval
pipeline steers with -- byte-identical conditioning, validated by the same
fingerprint rules (``read_ours_gcg_bag``), no SAE resident at fit time.

Bag derivation never needed direction artifacts, so the order is simply:

    ... stage 2 -> derive bags (SAE resident; adv run at magnitudes [0.0])
        -> fit encdec directions (encdec resident; bags READ from disk)
        -> counter-steering sweeps (reuse the same committed bags)

The bag digest is stamped into the direction artifact so a direction can never
be silently reused against a different bag.
"""

from __future__ import annotations

from pathlib import Path

from steering.eval.axbench_base.baselines.conditioning import EncoderConditioning
from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import read_ours_gcg_bag

# Directory name of the fair feature-DiffMean family inside an adv run's
# ``ours_gcg_bags`` store; one bag per (mode, concept) at each reg_lambda.
_FAIR_FEATURE_DIFFMEAN_DIRNAME = "feature_diffmean"


def load_encdec_fit_conditioning(
    bag_store: str | Path,
    *,
    mode: SteeringMode | str,
    concept_ids: list[int],
    expected_feature_contract_fingerprint: str | None,
) -> dict[int, EncoderConditioning]:
    """Load one committed fair feature-DiffMean bag per concept, for one mode.

    ``bag_store`` is an adv run's ``ours_gcg_bags`` directory. Every concept in
    ``concept_ids`` MUST resolve to exactly one committed bag; a missing bag,
    an ambiguous reg_lambda grid, an identity mismatch, or an empty bag RAISES.
    There is deliberately no fallback -- an empty or absent bag would silently
    reintroduce the unconditioned forward this exists to remove.
    """
    mode_value = SteeringMode(mode).value
    family_dir = Path(bag_store) / _FAIR_FEATURE_DIFFMEAN_DIRNAME / mode_value
    if not family_dir.is_dir():
        raise AxBenchBaseError(
            f"encdec fit conditioning: no fair feature-DiffMean bag directory at "
            f"{family_dir}; derive the bags first (an adv run at magnitudes "
            "[0.0] commits them) -- see #1249"
        )
    conditioning: dict[int, EncoderConditioning] = {}
    for concept_id in sorted(set(concept_ids)):
        concept_dir = family_dir / f"c{concept_id}"
        candidates = sorted(concept_dir.glob("lambda_*.safetensors"))
        if len(candidates) != 1:
            raise AxBenchBaseError(
                f"encdec fit conditioning: concept {concept_id} mode={mode_value} "
                f"has {len(candidates)} bag file(s) under {concept_dir} "
                f"({[c.name for c in candidates]}); exactly one committed bag is "
                "required -- a multi-lambda grid needs an explicit choice, and "
                "zero means the bags were never derived"
            )
        fid_to_bucket, metadata = read_ours_gcg_bag(
            candidates[0],
            expected_feature_contract_fingerprint=(
                expected_feature_contract_fingerprint
            ),
        )
        for key, expected in (
            ("concept_id", str(concept_id)),
            ("mode", mode_value),
            ("mode_tag", _FAIR_FEATURE_DIFFMEAN_DIRNAME),
        ):
            actual = metadata.get(key)
            if actual != expected:
                raise AxBenchBaseError(
                    f"encdec fit conditioning: bag {candidates[0]} carries "
                    f"{key}={actual!r}, expected {expected!r}; the bag store "
                    "layout disagrees with the artifacts' own identity"
                )
        if not fid_to_bucket:
            raise AxBenchBaseError(
                f"encdec fit conditioning: bag {candidates[0]} has no entries; "
                "an empty bag is the unconditioned forward #1249 removed"
            )
        conditioning[concept_id] = EncoderConditioning(
            entries=tuple(sorted((int(f), int(b)) for f, b in fid_to_bucket.items()))
        )
    return conditioning


__all__ = ["load_encdec_fit_conditioning"]
