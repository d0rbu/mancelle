"""Path helpers for axbench-bpb-adv artifacts."""

from __future__ import annotations

from pathlib import Path

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_base.paths import _sanitize
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    coerce_precomputed_bag_family,
)

SOURCE_CONTRACT_FILENAME = "source_contract.json"


def format_reg_lambda(reg_lambda: float) -> str:
    """Canonical, reverse-stable slug for a GCG ``reg_lambda`` value.

    Mirrors the repo-wide ``str(coef)`` coefficient convention (see
    ``axbench_bpb.bpb_scoring``): ``0.0 -> "0.0"``, ``0.5 -> "0.5"``,
    ``1.25 -> "1.25"``, ``1e-3 -> "0.001"``.  ``float(format_reg_lambda(x))``
    round-trips, so Task-7 bag loaders can reconstruct the exact path from the
    same ``(concept, mode, reg_lambda)`` triple.
    """
    return str(float(reg_lambda))


def format_objective_alpha(objective_alpha: float) -> str:
    """Canonical, reverse-stable slug for a class-margin ``objective_alpha``.

    Same contract as :func:`format_reg_lambda` — ``float(...)`` round-trips — so
    a bag loader reconstructs the exact path from the same
    ``(concept, mode, reg_lambda, alpha)`` tuple. Kept as its own name rather
    than reusing ``format_reg_lambda`` because the two slugs sit side by side in
    one filename and mean different things; a shared name there would invite
    passing one where the other belongs.
    """
    return str(float(objective_alpha))


class AxBenchBpbAdvPaths:
    def __init__(self, *, output_root: str | Path, run_tag: str) -> None:
        self.root = Path(output_root) / _sanitize(run_tag)

    def source_contract(self) -> Path:
        return self.root / SOURCE_CONTRACT_FILENAME

    def evaluation_population(self) -> Path:
        """Method-independent scored prefix/pair population for this run.

        Written by ``prepare_attack_rows`` and enforced by ``build_candidates`` /
        ``score``. It records the exact held-out ``core_prefix_id`` and
        rich/avoid completion-pair keys that every method family is allowed to
        evaluate.
        """
        return self.root / "evaluation_population.json"

    def attack_rows_dir(self) -> Path:
        return self.root / "attack_rows"

    def attack_rows_path(self) -> Path:
        return self.attack_rows_dir() / "attack_rows.parquet"

    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    def candidates_path(self) -> Path:
        return self.candidates_dir() / "candidates.parquet"

    def universal_gcg_checkpoint_dir(self) -> Path:
        """Immutable per-group Faster-GCG checkpoint root."""
        return self.candidates_dir() / "universal_gcg_checkpoint"

    def universal_gcg_checkpoint_manifest(self) -> Path:
        return self.universal_gcg_checkpoint_dir() / "manifest.json"

    def universal_gcg_checkpoint_groups_dir(self) -> Path:
        return self.universal_gcg_checkpoint_dir() / "groups"

    def context_cache_dir(self) -> Path:
        return self.root / "context_cache"

    def context_cache_file(self, prefix_hash: str) -> Path:
        return self.context_cache_dir() / prefix_hash[:2] / f"{prefix_hash}.safetensors"

    def context_cache_manifest(self) -> Path:
        return self.context_cache_dir() / "manifest.json"

    def clustered_context_cache_raw_dir(self) -> Path:
        """Clustered-only raw SAE evidence used for deterministic replay."""
        return self.context_cache_dir() / "raw"

    def clustered_context_cache_raw_file(self, prefix_hash: str) -> Path:
        """Raw per-prefix IDs/magnitudes written before clustered selection."""
        return (
            self.clustered_context_cache_raw_dir()
            / prefix_hash[:2]
            / f"{prefix_hash}.safetensors"
        )

    def clustered_context_cache_partial_marker(self) -> Path:
        """Incomplete-cache contract, including the pinned DDP world size."""
        return self.context_cache_dir() / "clustered_partial.json"

    def scores_path(self) -> Path:
        return self.root / "scores.parquet"

    def roundtrip_scores_path(self) -> Path:
        return self.root / "roundtrip_scores.parquet"

    def roundtrip_checkpoint_dir(self) -> Path:
        """Immutable per-candidate round-trip checkpoint root."""
        return self.root / "roundtrip_checkpoint"

    def roundtrip_checkpoint_manifest(self) -> Path:
        return self.roundtrip_checkpoint_dir() / "manifest.json"

    def general_val_bpb_path(self) -> Path:
        """Per-intervention general-validation BPB table (GENERAL_VAL_BPB_SCHEMA).

        Written by the optional ``score_general_val`` stage and left-joined onto
        the aggregates. Absent when ``general_val_bpb`` is disabled.
        """
        return self.root / "general_val_bpb.parquet"

    def general_val_bags_dir(self) -> Path:
        """Root for per-document roofline general-validation conditioning bags."""
        return self.root / "general_val_bags"

    def general_val_bag(
        self,
        sample_id: str,
        family: PrecomputedBagFamily | str,
        reg_lambda: float,
    ) -> Path:
        """Path to one concept-independent roofline general-val bag.

        These bags are keyed by validation ``sample_id`` rather than concept or
        training-prefix id. The roofline general-val oracle conditions each
        generic validation document on its own output-window signal, so there is
        no concept coordinate in the artifact path.
        """
        family_slug = coerce_precomputed_bag_family(family).value
        lam_slug = _sanitize(format_reg_lambda(reg_lambda))
        return (
            self.general_val_bags_dir()
            / _sanitize(family_slug)
            / f"s{_sanitize(str(sample_id))}"
            / f"lambda_{lam_slug}.safetensors"
        )

    def universal_triggers_path(self) -> Path:
        """Per-concept universal Faster-GCG trigger table (UNIVERSAL_TRIGGERS_SCHEMA).

        Written by the candidate stage (one row per non-control concept group):
        the shared k-token trigger optimised over the group's pairs. Read by
        ``general_val.py`` to substitute the trigger into the validation docs so
        ``general_val_bpb`` is measured under the attack. Always written (an empty
        trigger set still serializes a schema-stable table).
        """
        return self.root / "universal_triggers.parquet"

    def aggregates_dir(self) -> Path:
        return self.root / "aggregates"

    def roundtrip_aggregates_dir(self) -> Path:
        return self.root / "roundtrip_aggregates"

    def plots_dir(self) -> Path:
        return self.root / "plots"

    def roundtrip_plots_dir(self) -> Path:
        return self.root / "roundtrip_plots"

    def steering_strength_plots_dir(self) -> Path:
        return self.root / "steering_strength_plots"

    def samples_dir(self) -> Path:
        """Per-datapoint sampled-completion YAML docs (one tree per family).

        Written by the decoder-only completion generator
        (``sample_generation.run_generate_samples``): one YAML per steering
        family under ``samples/<mode>/<method>__<adv>.yaml``.
        """
        return self.root / "samples"

    def profile_dir(self) -> Path:
        """Where the optional `AxBenchProfiler` writes its outputs.

        Used by `scripts.benchmark.profile_axbench_bpb_adv` to bind a profiler
        around the production run. The production eval itself does NOT write
        here — its `profile_region`/`record_counter` calls are no-ops when no
        profiler is active.
        """
        return self.root / "profile"

    def ours_gcg_bags_dir(self) -> Path:
        """Root of the precomputed feature-conditioning bag artifacts.

        Kept as ``ours_gcg_bags`` for artifact compatibility. Written by the
        feature-GCG derives and by the scored feature-DiffMean derive, then read
        by the corresponding precomputed-bag partition modifiers at scoring time.
        """
        return self.root / "ours_gcg_bags"

    def ours_gcg_bag(
        self,
        concept_id: int,
        mode: SteeringMode | str,
        reg_lambda: float,
        family: PrecomputedBagFamily | str,
        prefix_id: str | None = None,
        objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
    ) -> Path:
        """Path to one precomputed feature-conditioning bag.

        The bag is keyed by ``(concept_id, mode, reg_lambda, family)`` — and,
        for per-pair roofline variants, an extra ``prefix_id``.
        ``family`` is coerced through :class:`PrecomputedBagFamily`; any other
        string RAISES ``ValueError`` rather
        than silently minting a new directory slug the loader would later report
        as a missing artifact.  The on-disk slugs are the enum values —
        byte-identical to the historical ``"fair"`` / ``"roofline"`` /
        ``"feature_diffmean"`` strings, plus ``"feature_diffmean_roofline"``.

        Layout::

            ours_gcg_bags/<family>/<mode>/c<concept_id>/lambda_<slug>.safetensors
            ours_gcg_bags/<family>/<mode>/c<concept_id>/p<prefix_id>/lambda_<slug>.safetensors

        The ``reg_lambda`` slug is the reverse-stable :func:`format_reg_lambda`
        value (sanitized), so a Task-7 loader reconstructs the same path from the
        same triple.

        ``objective_alpha`` appends an ``_alpha_<slug>`` suffix to the filename —
        but ONLY when it differs from the plain-margin default
        :data:`DEFAULT_OBJECTIVE_ALPHA`. At the default the path is byte-identical
        to what it was before the knob existed, which is the point: an alpha = 1
        derive produces arithmetically the same bag as a pre-alpha derive, so the
        two must resolve to the SAME artifact and reuse each other rather than
        re-running days of optimization. Non-default alphas land as siblings, so
        an alpha sweep never collides with the lambda sweep.
        """
        mode_value = SteeringMode(mode).value
        family_slug = coerce_precomputed_bag_family(family).value
        lam_slug = _sanitize(format_reg_lambda(reg_lambda))
        parent = (
            self.ours_gcg_bags_dir()
            / _sanitize(family_slug)
            / _sanitize(mode_value)
            / f"c{int(concept_id)}"
        )
        if prefix_id is not None:
            parent = parent / f"p{_sanitize(str(prefix_id))}"
        stem = f"lambda_{lam_slug}"
        if float(objective_alpha) != DEFAULT_OBJECTIVE_ALPHA:
            stem = f"{stem}_alpha_{_sanitize(format_objective_alpha(objective_alpha))}"
        return parent / f"{stem}.safetensors"

    def ours_gcg_scored_prefixes(
        self,
        concept_id: int,
        mode: SteeringMode | str,
        family: PrecomputedBagFamily | str,
    ) -> Path:
        """Path to the per-(concept, mode, family) SCORED-prefix artifact.

        Precomputed-bag derives persist the ``core_prefix_id``s the score stage is
        allowed to score for this group: fair GCG and fair feature-DiffMean
        record HELD-OUT prefixes (clean generalization), while roofline GCG and
        roofline feature-DiffMean record that same held-out prefix set but derive
        one oracle bag per held-out prefix. The score stage drops any bag-backed
        candidate row whose ``core_prefix_id`` is not in this set, so plotted BPB
        is measured on exactly the intended prefixes (and a scored prefix with no
        bag is then a genuine bug -> loud raise). ``family`` is coerced through
        :class:`PrecomputedBagFamily`; any other string RAISES ``ValueError``.

        Layout (a sibling of the bag tree)::

            ours_gcg_bags/<family>/<mode>/c<concept_id>/scored_prefixes.json
        """
        mode_value = SteeringMode(mode).value
        return (
            self.ours_gcg_bags_dir()
            / _sanitize(coerce_precomputed_bag_family(family).value)
            / _sanitize(mode_value)
            / f"c{int(concept_id)}"
            / "scored_prefixes.json"
        )
